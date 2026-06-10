from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


class BaseModelSetup(
    TimedActionMixin,
    metaclass=ABCMeta,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode
        self.frozen_parameters = {}
        self._dpo_ref_params = None
        self._last_dpo_metrics = None
        self._dpo_paired_half = None  # read by ModelSetupNoiseMixin._apply_dpo_paired_rng
        self._dpo_runtime_beta = None

    @abstractmethod
    def create_parameters(
            self,
            model: BaseModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        pass

    @abstractmethod
    def setup_optimizations(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_model(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_train_device(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass

    def report_to_tensorboard(
            self,
            model: BaseModel,
            config: TrainConfig,
            scheduler: LRScheduler,
            tensorboard: SummaryWriter,
    ):
        lrs = scheduler.get_last_lr()
        parameters = model.parameters.display_name_mapping

        reported_learning_rates = {}

        # Handle MuonWithAuxAdam's split parameter groups
        if any('optim_type' in g for g in model.optimizer.param_groups):
            for group in model.optimizer.param_groups:
                name = group.get('name')
                if not name or not group['params']:
                    continue
                # For MuonWithAuxAdam, parameter groups are split for Muon and Adam,
                # but might retain the same base name (e.g., 'unet').
                optim_type = group.get('optim_type', 'unknown')
                unique_name = f"{name}_{optim_type}"
                if unique_name not in reported_learning_rates:
                    reported_learning_rates[unique_name] = group['lr']
        else:
            for lr, parameter in zip(lrs, parameters, strict=True):
                # only use the prefix. this prevents multiple embedding reports. TODO: find a better solution
                name = parameter.split('/')[0]

                if name not in reported_learning_rates:
                    reported_learning_rates[name] = lr

        reported_learning_rates = config.optimizer.optimizer.maybe_adjust_lrs(reported_learning_rates, model.optimizer)

        for name, lr in reported_learning_rates.items():
            tensorboard.add_scalar(
                f"lr/{name}", lr, model.train_progress.global_step
            )

        if hasattr(model.optimizer, 'kourkoutas_helper') and model.optimizer.kourkoutas_helper is not None:
            stats = model.optimizer.kourkoutas_helper.last_beta2_stats
            if stats:
                tensorboard.add_scalar("kourkoutas/beta2_mean", stats['mean'], model.train_progress.global_step)

    @staticmethod
    def _is_dpo_rejected_key(key: str) -> bool:
        return key.endswith("_rejected")

    @classmethod
    def _create_dpo_batched_batch(cls, batch: dict) -> tuple[dict, int]:
        # Returns a batch where every <key>/<key>_rejected pair is concatenated
        # as [chosen; rejected] on dim 0, shared per-sample tensors are duplicated
        # by self-concat, and non-batched values pass through. The chosen half is
        # always the first B entries of the result.
        chosen_b = batch["latent_image"].shape[0]
        batched: dict = {}
        for key, value in batch.items():
            if cls._is_dpo_rejected_key(key):
                continue
            rejected_key = key + "_rejected"
            if rejected_key in batch:
                batched[key] = torch.cat([value, batch[rejected_key]], dim=0)
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == chosen_b:
                batched[key] = torch.cat([value, value], dim=0)
            else:
                batched[key] = value
        return batched, chosen_b

    @staticmethod
    def _split_dpo_batched_output(output: dict, chosen_b: int) -> tuple[dict, dict]:
        # Splits a model output dict whose batched tensors have leading dim 2B
        # into chosen-only (first B) and rejected-only (last B) dicts.
        chosen_out: dict = {}
        rejected_out: dict = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2 * chosen_b:
                chosen_out[key] = value[:chosen_b]
                rejected_out[key] = value[chosen_b:]
            else:
                chosen_out[key] = value
                rejected_out[key] = value
        return chosen_out, rejected_out

    def get_last_dpo_metrics(self) -> dict[str, float]:
        return self._last_dpo_metrics or {}

    def set_dpo_runtime_beta(self, beta: float | None):
        # Adaptive-beta override from the trainer. The logged reward metrics
        # are computed before beta is applied, so adapting beta from them does
        # not create a feedback loop.
        self._dpo_runtime_beta = beta

    def calculate_dpo_loss(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
    ) -> Tensor:
        if "latent_image_rejected" not in batch:
            raise RuntimeError(
                "RLHF DPO requires paired chosen/rejected batches, but the dataloader did not provide rejected samples."
            )

        def mse_per_sample(pred, target):
            # fp32 accumulation in the reduction instead of upcasting the full
            # [2B,C,H,W] tensors - avoids four fp32 copies per step.
            return (pred - target).pow(2).mean(dim=list(range(1, pred.ndim)), dtype=torch.float32)

        beta = config.rlhf_dpo_beta if self._dpo_runtime_beta is None else self._dpo_runtime_beta
        supervised_loss = None

        # 2 forwards: 1 batched ref (no_grad) + 1 batched policy, each over the
        # [chosen; rejected] batch. Both halves share per-pair timestep+noise via
        # _dpo_paired_half, and ref/policy share them too because predict()
        # seeds its generator from global_step. Note for torch.compile users:
        # supervised/validation batches are B-sized while DPO batches are
        # 2B-sized, so mixing them in one session compiles two graphs.
        batched_input, chosen_b = self._create_dpo_batched_batch(batch)

        self._dpo_paired_half = chosen_b
        try:
            with torch.no_grad(), self.reference_model(model, config):
                ref_output = self.predict(model, batched_input, config, train_progress)
                ref_predicted = ref_output["predicted"]
                ref_target = ref_output["target"]
                ref_chosen_logp = -mse_per_sample(ref_predicted[:chosen_b], ref_target[:chosen_b])
                ref_rejected_logp = -mse_per_sample(ref_predicted[chosen_b:], ref_target[chosen_b:])
                del ref_output, ref_predicted, ref_target

            policy_output = self.predict(model, batched_input, config, train_progress)
        finally:
            self._dpo_paired_half = None
        policy_timestep = policy_output.get("timestep")
        policy_predicted = policy_output["predicted"]
        policy_target = policy_output["target"]
        policy_chosen_logp = -mse_per_sample(policy_predicted[:chosen_b], policy_target[:chosen_b])
        policy_rejected_logp = -mse_per_sample(policy_predicted[chosen_b:], policy_target[chosen_b:])
        if config.rlhf_supervised_mix > 0:
            chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
            supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
            del chosen_output
        del policy_output, policy_predicted, policy_target

        chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
        rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
        margin = chosen_ratio - rejected_ratio

        if config.rlhf_dpo_objective == DPOObjective.IPO:
            # IPO regresses the raw margin toward the fixed target 1/(2*tau)
            # instead of pushing it to infinity, which structurally resists
            # reward hacking. tau plays beta's role; label smoothing and beta
            # do not apply.
            dpo_loss = (margin - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)).pow(2).mean()
            loss = dpo_loss
        else:
            logits = beta * margin
            dpo_loss = -F.logsigmoid(logits).mean()
            loss = dpo_loss

            if config.rlhf_dpo_label_smoothing > 0:
                s = config.rlhf_dpo_label_smoothing
                loss = (1 - s) * loss + s * (-F.logsigmoid(-logits).mean())

        if supervised_loss is not None:
            loss = loss + config.rlhf_supervised_mix * supervised_loss
            del supervised_loss

        self._last_dpo_metrics = {
            "loss": loss.detach().item(),
            "dpo_loss": dpo_loss.detach().item(),
            "chosen_reward": chosen_ratio.detach().mean().item(),
            "rejected_reward": rejected_ratio.detach().mean().item(),
            "reward_margin": margin.detach().mean().item(),
            "accuracy": (chosen_ratio > rejected_ratio).float().mean().item(),
        }

        if config.rlhf_dpo_timestep_margin_logging and policy_timestep is not None:
            # Per-sample raw margins bucketed by the chosen half's timestep
            # quartile. Sums and counts are emitted for every quartile so the
            # trainer's accumulation always sees the same key set.
            t = policy_timestep[:chosen_b].detach().float()
            if t.numel() > 0 and t.max() > 1.0:
                t = t / 1000.0  # discrete schedulers train on 1000 timesteps
            quartile_index = (t * 4).long().clamp(0, 3)
            per_sample_margin = margin.detach()
            for quartile in range(4):
                mask = quartile_index == quartile
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_sum"] = per_sample_margin[mask].sum().item()
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_count"] = float(mask.sum().item())

        return loss

    def stop_embedding_training_elapsed(
            self,
            config: TrainEmbeddingConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training_" + str(config.uuid),
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    def __stop_model_part_training_elapsed(
            self,
            unique_name: str,
            config: TrainModelPartConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_" + unique_name + "_training",
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    @contextmanager
    def prior_model(self, model: BaseModel, config: TrainConfig):
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Prior model is only available with LoRA training")

        for adapter in model.adapters():
            adapter.remove_hook_from_module()
        try:
            yield
        finally:
            for adapter in model.adapters():
                adapter.hook_to_module()

    @contextmanager
    def reference_model(self, model: BaseModel, config: TrainConfig):
        adapters = model.adapters()

        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError(
                "RLHF DPO reference modes are currently only implemented for adapter training in the LoRA tab."
            )
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO requires active adapters, but no trainable adapters are attached to the current model."
            )

        ref_mode = config.effective_dpo_ref_mode()

        if ref_mode == DPORefMode.NEW_ADAPTER:
            for adapter in adapters:
                adapter.remove_hook_from_module()
            try:
                yield
            finally:
                for adapter in adapters:
                    adapter.hook_to_module()
        elif ref_mode == DPORefMode.EXISTING_ADAPTER:
            # Reference params are captured once at first call and frozen for the
            # entire training run. This is intentional: DPO requires a fixed
            # reference policy from the start of training.
            if self._dpo_ref_params is None:
                self._dpo_ref_params = [[p.data.clone() for p in adapter.parameters()] for adapter in adapters]

            policy_data = [[p.data for p in adapter.parameters()] for adapter in adapters]
            try:
                for adapter, ref_params in zip(adapters, self._dpo_ref_params, strict=True):
                    for param, ref_data in zip(adapter.parameters(), ref_params, strict=True):
                        param.data = ref_data
                yield
            finally:
                for adapter, policy_ptrs in zip(adapters, policy_data, strict=True):
                    for param, policy_ptr in zip(adapter.parameters(), policy_ptrs, strict=True):
                        param.data = policy_ptr
        else:
            raise ValueError(f"Unsupported DPO reference mode: {ref_mode}")

    def _create_model_part_parameters(
        self,
        parameter_group_collection: NamedParameterGroupCollection,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        freeze: list[ModuleFilter] | None = None,
        debug: bool = False,
    ):
        if not config.train:
            return

        if freeze is not None and len(freeze) > 0:
            selected = []
            deselected = []
            parameters = []
            self.frozen_parameters[unique_name] = []
            for name, param in model.named_parameters():
                if any(f.matches(name) for f in freeze):
                    parameters.append(param)
                    selected.append(name)
                else:
                    self.frozen_parameters[unique_name].append(param)
                    deselected.append(name)

            if debug:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")
        else:
            parameters = model.parameters()

        parameter_group_collection.add_group(NamedParameterGroup(
            unique_name=unique_name,
            parameters=parameters,
            learning_rate=config.learning_rate,
        ))

    def _setup_model_part_requires_grad(
        self,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        train_progress: TrainProgress,
    ):
        if model is not None:
            train_model_part = config.train and \
                               not self.__stop_model_part_training_elapsed(unique_name, config, train_progress)
            model.requires_grad_(train_model_part)

            #even if frozen parameters are not passed to the optimizer, required_grad has to be False.
            #otherwise, gradients accumulate in param.grad and waste vram
            if unique_name in self.frozen_parameters:
                for param in self.frozen_parameters[unique_name]:
                    param.requires_grad_(False)
