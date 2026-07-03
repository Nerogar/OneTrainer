import os
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
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == chosen_b for v in value)
            ):
                # SDXL-style per-field tensor lists (original_resolution / crop_resolution /
                # crop_offset collate a per-sample tuple into a list of [B] tensors). The
                # rejected image shares the chosen sample's bucket, so duplicate each field
                # elementwise to reach 2B and keep add_time_ids aligned with the latents.
                batched[key] = type(value)(torch.cat([v, v], dim=0) for v in value)
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

        # Conditioning (caption) dropout is drawn per-sample inside encode_text.
        # On the [chosen; rejected] batch that independently zeroes the prompt on
        # one half of a pair (~2p(1-p) of pairs), comparing a prompted sample
        # against an unconditional one and silently corrupting the preference
        # margin. Neutralize it for the paired forward so both halves see
        # identical conditioning; restore afterwards. (Diffusion-DPO trains on the
        # given prompts, not CFG-dropped ones.)
        te_configs = (config.text_encoder, config.text_encoder_2, config.text_encoder_3, config.text_encoder_4)
        saved_dropout = [te.dropout_probability for te in te_configs]

        self._dpo_paired_half = chosen_b
        try:
            for te in te_configs:
                te.dropout_probability = 0.0
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
            for te, saved in zip(te_configs, saved_dropout, strict=True):
                te.dropout_probability = saved
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

        # Stack the six scalar metrics and sync once (six separate .item() calls
        # would be six GPU->CPU stalls on the training hot path).
        metric_values = torch.stack([
            loss.detach().float(),
            dpo_loss.detach().float(),
            chosen_ratio.detach().mean(),
            rejected_ratio.detach().mean(),
            margin.detach().mean(),
            (chosen_ratio > rejected_ratio).float().mean(),
        ]).tolist()
        self._last_dpo_metrics = dict(zip(
            ("loss", "dpo_loss", "chosen_reward", "rejected_reward", "reward_margin", "accuracy"),
            metric_values,
            strict=True,
        ))

        if config.rlhf_dpo_timestep_margin_logging and policy_timestep is not None:
            # Per-sample raw margins bucketed by the chosen half's timestep
            # quartile. Sums and counts are emitted for every quartile so the
            # trainer's accumulation always sees the same key set.
            raw_t = policy_timestep[:chosen_b].detach()
            if torch.is_floating_point(raw_t):
                # Continuous-timestep models already emit t in (0, 1].
                t = raw_t.float()
            else:
                # Discrete schedulers emit integer steps in [0, num_train_timesteps);
                # normalize by the model's actual count rather than sniffing the
                # batch max (which misbuckets batches that only sample steps {0,1})
                # or assuming 1000.
                num_train_timesteps = model.noise_scheduler.config['num_train_timesteps']
                t = raw_t.float() / num_train_timesteps
            quartile_index = (t * 4).long().clamp(0, 3)
            per_sample_margin = margin.detach().float()
            # bincount for counts + scatter_add for sums, then one sync for all
            # eight quartile scalars instead of eight .item() calls.
            counts = torch.bincount(quartile_index, minlength=4).float()
            sums = torch.zeros(4, device=per_sample_margin.device, dtype=per_sample_margin.dtype)
            sums.scatter_add_(0, quartile_index, per_sample_margin)
            sums_l, counts_l = torch.stack([sums, counts]).tolist()
            for quartile in range(4):
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_sum"] = sums_l[quartile]
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_count"] = counts_l[quartile]

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
                    for i, (param, ref_data) in enumerate(zip(adapter.parameters(), ref_params, strict=True)):
                        # Reference params restored from a backup live on CPU; move
                        # them onto the live param's device/dtype once and cache the
                        # moved tensor (a no-op for the lazily-cloned in-run case).
                        if ref_data.device != param.data.device or ref_data.dtype != param.data.dtype:
                            ref_data = ref_data.to(device=param.data.device, dtype=param.data.dtype)
                            ref_params[i] = ref_data
                        param.data = ref_data
                yield
            finally:
                for adapter, policy_ptrs in zip(adapters, policy_data, strict=True):
                    for param, policy_ptr in zip(adapter.parameters(), policy_ptrs, strict=True):
                        param.data = policy_ptr
        else:
            raise ValueError(f"Unsupported DPO reference mode: {ref_mode}")

    _DPO_REFERENCE_FILENAME = "dpo_reference.pt"

    def save_dpo_reference(self, backup_path: str) -> None:
        # Persist the frozen EXISTING_ADAPTER reference alongside a backup so it
        # survives resume. Without this the reference is re-captured from the
        # resumed (already DPO-trained) adapter weights, silently moving the KL
        # anchor. NEW_ADAPTER mode never captures params, so nothing is written.
        if self._dpo_ref_params is None:
            return
        try:
            payload = [[t.detach().to("cpu", copy=True) for t in adapter] for adapter in self._dpo_ref_params]
            torch.save(payload, os.path.join(backup_path, self._DPO_REFERENCE_FILENAME))
        except OSError:
            pass

    def load_dpo_reference(self, backup_path: str, model: BaseModel) -> bool:
        # Restore the frozen reference saved next to the backup we are resuming
        # from. Returns True on success; on any structural mismatch it leaves
        # _dpo_ref_params unset so the caller falls back to lazy capture rather
        # than adopting a wrong reference.
        path = os.path.join(backup_path, self._DPO_REFERENCE_FILENAME)
        if not os.path.isfile(path):
            return False
        saved = torch.load(path, map_location="cpu", weights_only=True)
        adapters = model.adapters()
        if not isinstance(saved, list) or len(saved) != len(adapters):
            return False
        restored: list[list[Tensor]] = []
        for adapter, adapter_saved in zip(adapters, saved, strict=True):
            params = list(adapter.parameters())
            if len(adapter_saved) != len(params):
                return False
            if any(tuple(t.shape) != tuple(p.shape) for t, p in zip(adapter_saved, params, strict=True)):
                return False
            restored.append(list(adapter_saved))
        self._dpo_ref_params = restored
        return True

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
