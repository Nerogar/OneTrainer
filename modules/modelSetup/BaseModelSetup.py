from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.enum.DPOExecutionMode import DPOExecutionMode
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
    def _create_dpo_rejected_batch(cls, batch: dict) -> dict:
        rejected_batch = dict(batch)
        for key, value in batch.items():
            if cls._is_dpo_rejected_key(key):
                rejected_batch[key.removesuffix("_rejected")] = value
        return rejected_batch

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

    @staticmethod
    def _create_dpo_progress(train_progress: TrainProgress, offset: int = 0) -> TrainProgress:
        return TrainProgress(
            epoch=train_progress.epoch,
            epoch_step=train_progress.epoch_step,
            epoch_sample=train_progress.epoch_sample,
            global_step=train_progress.global_step + offset,
        )

    def get_last_dpo_metrics(self) -> dict[str, float]:
        return self._last_dpo_metrics or {}

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

        rejected_batch = self._create_dpo_rejected_batch(batch)
        chosen_progress = self._create_dpo_progress(train_progress)
        rejected_progress = (
            chosen_progress if config.rlhf_dpo_shared_noise else self._create_dpo_progress(train_progress, offset=1)
        )  # Shared noise depends on predict() seeding from global_step.

        def mse_per_sample(pred, target):
            return ((pred - target) ** 2).mean(dim=list(range(1, pred.ndim)))

        beta = config.rlhf_dpo_beta
        execution_mode = getattr(config, "rlhf_dpo_execution_mode", DPOExecutionMode.FULL_CONCURRENT)
        supervised_loss = None

        # Batching chosen+rejected into a single forward (along dim 0) requires
        # both halves to share timestep+noise, which predict() seeds from
        # global_step. When shared_noise is off, chosen runs at global_step and
        # rejected at global_step+1, so they cannot share a forward.
        can_batch = config.rlhf_dpo_shared_noise

        if can_batch and execution_mode == DPOExecutionMode.FULL_CONCURRENT:
            # 2 forwards: 1 batched ref (no_grad) + 1 batched policy.
            batched_input, chosen_b = self._create_dpo_batched_batch(batch)

            self._dpo_paired_half = chosen_b
            try:
                with torch.no_grad(), self.reference_model(model, config):
                    ref_output = self.predict(model, batched_input, config, chosen_progress)
                    ref_predicted = ref_output["predicted"].float()
                    ref_target = ref_output["target"].float()
                    ref_chosen_logp = -mse_per_sample(ref_predicted[:chosen_b], ref_target[:chosen_b])
                    ref_rejected_logp = -mse_per_sample(ref_predicted[chosen_b:], ref_target[chosen_b:])
                    del ref_output, ref_predicted, ref_target

                policy_output = self.predict(model, batched_input, config, chosen_progress)
            finally:
                self._dpo_paired_half = None
            policy_predicted = policy_output["predicted"].float()
            policy_target = policy_output["target"].float()
            policy_chosen_logp = -mse_per_sample(policy_predicted[:chosen_b], policy_target[:chosen_b])
            policy_rejected_logp = -mse_per_sample(policy_predicted[chosen_b:], policy_target[chosen_b:])
            if config.rlhf_supervised_mix > 0:
                chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
                supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
                del chosen_output
            del policy_output, policy_predicted, policy_target

        elif can_batch and execution_mode == DPOExecutionMode.POLICY_CONCURRENT:
            # 3 forwards: 2 ref serial (no_grad) + 1 batched policy.
            with torch.no_grad(), self.reference_model(model, config):
                ref_chosen_output = self.predict(model, batch, config, chosen_progress)
                ref_chosen_logp = -mse_per_sample(
                    ref_chosen_output["predicted"].float(), ref_chosen_output["target"].float()
                )
                del ref_chosen_output
                ref_rejected_output = self.predict(model, rejected_batch, config, rejected_progress)
                ref_rejected_logp = -mse_per_sample(
                    ref_rejected_output["predicted"].float(), ref_rejected_output["target"].float()
                )
                del ref_rejected_output

            batched_input, chosen_b = self._create_dpo_batched_batch(batch)
            self._dpo_paired_half = chosen_b
            try:
                policy_output = self.predict(model, batched_input, config, chosen_progress)
            finally:
                self._dpo_paired_half = None
            policy_predicted = policy_output["predicted"].float()
            policy_target = policy_output["target"].float()
            policy_chosen_logp = -mse_per_sample(policy_predicted[:chosen_b], policy_target[:chosen_b])
            policy_rejected_logp = -mse_per_sample(policy_predicted[chosen_b:], policy_target[chosen_b:])
            if config.rlhf_supervised_mix > 0:
                chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
                supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
                del chosen_output
            del policy_output, policy_predicted, policy_target

        elif execution_mode == DPOExecutionMode.FULL_CONCURRENT:
            # Fallback (shared_noise=False) — preserves original 4-forward FULL_CONCURRENT memory schedule.
            with torch.no_grad(), self.reference_model(model, config):
                ref_chosen_output = self.predict(model, batch, config, chosen_progress)
                ref_rejected_output = self.predict(model, rejected_batch, config, rejected_progress)

            policy_chosen_output = self.predict(model, batch, config, chosen_progress)
            policy_rejected_output = self.predict(model, rejected_batch, config, rejected_progress)

            ref_chosen_logp = -mse_per_sample(
                ref_chosen_output["predicted"].float(), ref_chosen_output["target"].float()
            )
            ref_rejected_logp = -mse_per_sample(
                ref_rejected_output["predicted"].float(), ref_rejected_output["target"].float()
            )
            policy_chosen_logp = -mse_per_sample(
                policy_chosen_output["predicted"].float(), policy_chosen_output["target"].float()
            )
            if config.rlhf_supervised_mix > 0:
                supervised_loss = self.calculate_loss(model, batch, policy_chosen_output, config)
            policy_rejected_logp = -mse_per_sample(
                policy_rejected_output["predicted"].float(), policy_rejected_output["target"].float()
            )
            del ref_chosen_output
            del ref_rejected_output
            del policy_chosen_output
            del policy_rejected_output
        else:
            # SEQUENTIAL (any noise setting), or POLICY_CONCURRENT fallback when shared_noise=False.
            with torch.no_grad(), self.reference_model(model, config):
                ref_chosen_output = self.predict(model, batch, config, chosen_progress)
                ref_chosen_logp = -mse_per_sample(
                    ref_chosen_output["predicted"].float(), ref_chosen_output["target"].float()
                )
                del ref_chosen_output
                ref_rejected_output = self.predict(model, rejected_batch, config, rejected_progress)
                ref_rejected_logp = -mse_per_sample(
                    ref_rejected_output["predicted"].float(), ref_rejected_output["target"].float()
                )
                del ref_rejected_output

            policy_chosen_output = self.predict(model, batch, config, chosen_progress)
            policy_chosen_logp = -mse_per_sample(
                policy_chosen_output["predicted"].float(), policy_chosen_output["target"].float()
            )
            if config.rlhf_supervised_mix > 0:
                supervised_loss = self.calculate_loss(model, batch, policy_chosen_output, config)
            if execution_mode == DPOExecutionMode.SEQUENTIAL:
                del policy_chosen_output

            policy_rejected_output = self.predict(model, rejected_batch, config, rejected_progress)
            policy_rejected_logp = -mse_per_sample(
                policy_rejected_output["predicted"].float(), policy_rejected_output["target"].float()
            )
            if execution_mode != DPOExecutionMode.SEQUENTIAL:
                del policy_chosen_output
            del policy_rejected_output

        chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
        rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
        logits = beta * (chosen_ratio - rejected_ratio)
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
            "reward_margin": (chosen_ratio - rejected_ratio).detach().mean().item(),
            "accuracy": (chosen_ratio > rejected_ratio).float().mean().item(),
        }

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
