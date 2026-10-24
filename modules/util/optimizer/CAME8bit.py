#
# Copied and modified from the original CAME implementation (https://github.com/yangluo7/CAME)
#
# Implements the "Memory-Efficient CAME-8bit Optimizer" modifications described in the paper:
#   "SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear
#    Diffusion Transformer" (https://arxiv.org/pdf/2501.18427v3)
# Implements cautious masking from "Cautious Optimizers: Improving Training with One Line of Code"
#    (https://arxiv.org/abs/2411.16085)

from modules.util.bf16_stochastic_rounding import add_stochastic_
from modules.util.torch_util import torch_gc

import torch
import torch.optim


class CAME8bit(torch.optim.Optimizer):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization` and
    `Memory-Efficient CAME-8bit Optimizer`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
        update, square gradient and instability (default: (0.9, 0.999, 0.9999))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        stochastic_rounding: utilize stochastic rounding with BF16 on non-8bit params (default: False)
        use_cautious: use cautious masking (default: False)
        min_8bit_size (int) The minimum size of a tensor before it is eligible to be quantized (default: 16384)
        quant_block_size (int) The amount of values to quantize into a single block (default: 2048)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        stochastic_rounding=False,
        use_cautious=False,
        min_8bit_size=16384,
        quant_block_size=2048,
    ):
        assert lr > 0.
        assert all(0. <= beta <= 1. for beta in betas)

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "betas": betas,
            "weight_decay": weight_decay,
            "stochastic_rounding": stochastic_rounding,
            "use_cautious": use_cautious,
            "min_8bit_size": min_8bit_size,
            "quant_block_size": quant_block_size,
        }
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        # We will convert our data to/from float32
        # instead of having that done for us.
        return True

    @property
    def supports_flat_params(self):
        # We do not support using a single contiguous
        # Tensor for all of the parameters.
        return False

    @staticmethod
    def _should_use_matrix_factorization(grad_shape: torch.Size):
        grad_shape_dimensions = len(grad_shape)
        return grad_shape_dimensions == 2 or \
                (grad_shape_dimensions == 4 and grad_shape[2] == 1 and grad_shape[3] == 1)

    @staticmethod
    def _should_quantize_param(grad_shape: torch.Size, min_8bit_size: int):
        # We want to quantize blocks that have larger than `min_8bit_size`, but
        # only if they are `linear` or `1x1 convolution` layers. see: "Block-wise
        # Quantization Strategy", arxiv:2501.18427v3, section 2.3.
        #
        # Note: Matrixes must be _greater_ than `min_8bit_size` in size to be quantized,
        #       which may be different than what you might expect. In other words, the
        #       `>` below is intended and should not be `>=`.

        if CAME8bit._should_use_matrix_factorization(grad_shape):
            return grad_shape.numel() > min_8bit_size

        return False

    @staticmethod
    def _quantize_param(params: torch.Tensor, quant_block_size: int):
        # Quantize our values in normalized `quant_block_size`-sized blocks.
        # We shift the range of our data between max/min, so the closer data values
        # are to each other, the higher precision our data is. The value of
        # `quant_block_size` should balance space-savings and data precision.

        if params.numel() <= 1:
            return params

        data_chunk_list = params.split(quant_block_size)
        quantized_values: list = [None] * len(data_chunk_list)
        for index, data_chunk in enumerate(data_chunk_list, start=0):
            max_value = data_chunk.max()
            min_value = data_chunk.min()
            normalize_scale = (max_value - min_value) / 255.0

            values = ((data_chunk - min_value) / normalize_scale).round().byte()

            quantized_values[index] = {"value": values, "scale": normalize_scale, "min": min_value}

        return quantized_values

    @staticmethod
    def _dequantize_param(quantized_value_list):
        # If this isn't a quantized list, give it back
        if not isinstance(quantized_value_list, list):
            return quantized_value_list

        dequantized_values: list = [None] * len(quantized_value_list)
        for index, quantized_chunk in enumerate(quantized_value_list, start=0):
            dequantized_values[index] = \
                (quantized_chunk["value"].float() * quantized_chunk["scale"]) + quantized_chunk["min"]

        return torch.cat(dequantized_values)

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        grad = p.grad.data

        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        if grad.is_sparse:
            raise RuntimeError("CAME_8BIT does not support sparse gradients.")

        grad_shape = grad.shape

        use_factor = CAME8bit._should_use_matrix_factorization(grad_shape)
        should_quantize_param = CAME8bit._should_quantize_param(grad_shape, group["min_8bit_size"])

        state = self.state[p]

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            state["RMS"] = 0

            state["exp_avg"] = torch.zeros_like(grad) if not should_quantize_param else \
                               CAME8bit._quantize_param(torch.zeros_like(grad), group["quant_block_size"])

            if use_factor:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[1]).type_as(grad)

                state["exp_avg_res_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[1]).type_as(grad)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad) if not should_quantize_param else \
                                      CAME8bit._quantize_param(torch.zeros_like(grad), group["quant_block_size"])

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        update: torch.Tensor = (grad**2) + group["eps"][0]
        if use_factor:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]

            # Trim off extra 1 dimensions if our tensor isn't 2D
            sq_update = update if len(grad_shape) == 2 else update.squeeze()

            # Do update
            exp_avg_sq_row.mul_(group["betas"][1]).add_(
                sq_update.mean(dim=-1), alpha=1.0 - group["betas"][1]
            )
            exp_avg_sq_col.mul_(group["betas"][1]).add_(
                sq_update.mean(dim=-2), alpha=1.0 - group["betas"][1]
            )

            # Approximation of exponential moving average of square of gradient
            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)

            # Convert a trimmed tensor back into the original shape
            if update.shape != grad_shape:
                update = update.view(grad_shape)
        else:
            # Dequantize if needed
            exp_avg_sq: torch.Tensor = state["exp_avg_sq"] if not should_quantize_param else \
                                       CAME8bit._dequantize_param(state["exp_avg_sq"])

            # Do update
            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
            update = exp_avg_sq.rsqrt()

            # Requantize if needed
            state["exp_avg_sq"] = exp_avg_sq if not should_quantize_param else \
                                  CAME8bit._quantize_param(exp_avg_sq, group["quant_block_size"])

        update.mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # Dequantize if needed
        exp_avg = state["exp_avg"] if not should_quantize_param else \
                  CAME8bit._dequantize_param(state["exp_avg"])

        # Do update
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

        # Requantize if needed
        state["exp_avg"] = exp_avg if not should_quantize_param else \
                           CAME8bit._quantize_param(exp_avg, group["quant_block_size"])

        # Confidence-guided strategy
        # Calculation of instability
        if use_factor:
            res = (update - exp_avg)**2 + group["eps"][1]

            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]

            # Trim off extra 1 dimensions if our tensor isn't 2D
            re_update = res if len(grad_shape) == 2 else res.squeeze()

            # Do update
            exp_avg_res_row.mul_(group["betas"][2]).add_(
                re_update.mean(dim=-1), alpha=1.0 - group["betas"][2]
            )
            exp_avg_res_col.mul_(group["betas"][2]).add_(
                re_update.mean(dim=-2), alpha=1.0 - group["betas"][2]
            )

            # Approximation of exponential moving average of instability
            res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)

            # Convert a trimmed tensor back into the original shape
            if res_approx.shape != grad.shape:
                res_approx = res_approx.view(grad.shape)

            update = res_approx.mul_(exp_avg)
        else:
            update = exp_avg.clone()

        # Cautious masking (https://arxiv.org/abs/2411.16085)
        if group["use_cautious"]:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)

        # Decay weights if needed
        if group["weight_decay"] > 0:
            if p.dtype == torch.bfloat16 and group["stochastic_rounding"]:
                add_stochastic_(p.data, p.data,
                                alpha=-group["weight_decay"] * group["lr"])
            else:
                p.data.add_(
                    p.data, alpha=-group["weight_decay"] * group["lr"]
                )

        # Write update
        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and group["stochastic_rounding"]:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        return loss

    def load_state_dict(self, state_dict):
        # Load the model's data
        super().load_state_dict(state_dict)

        # Reinitialize existing quantized values (lists of objects) as a byte
        # Reinitialize existing unquantized values (tensors) as a float
        quantizable_value_keys = [
            "exp_avg",
            "exp_avg_sq",
            "exp_avg_sq_col", "exp_avg_sq_row",
            "exp_avg_res_col", "exp_avg_res_row",
        ]
        for state in self.state.values():
            for quant_state_key in quantizable_value_keys:
                if quant_state_key in state:
                    if isinstance(state[quant_state_key], list):
                        for quantized_object in state[quant_state_key]:
                            quantized_object["value"] = quantized_object["value"].byte()
                    elif isinstance(state[quant_state_key], torch.Tensor):
                        state[quant_state_key] = state[quant_state_key].float()

        # Not sure if this is GC is needed, but the reference implementation had one
        torch_gc()
