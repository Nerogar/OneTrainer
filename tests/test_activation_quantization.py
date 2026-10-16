import tempfile
import unittest
from pathlib import Path

from modules.modelLoader.mixin.HFModelLoaderMixin import stream_module_from_checkpoint
from modules.module.quantized.LinearA8 import LinearA8
from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.ui.BaseModelTabView import BaseModelTabView
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.disk_stream import evict_to_meta
from modules.util.enum.DataType import DataType
from modules.util.quantization_util import (
    is_quantized_parameter,
    quantize_layers,
    replace_linear_with_quantized_layers,
)

import torch
from torch import nn

from safetensors.torch import save_file

A8_DTYPES = (
    (DataType.BFLOAT_16_A8_INT, torch.int8),
    (DataType.BFLOAT_16_A8_FLOAT, torch.float8_e4m3fn),
)


class ActivationQuantizationTest(unittest.TestCase):
    def test_a8_and_compression_dropdown_options(self):
        for include_a8 in (False, True):
            for include_compressed in (False, True):
                with self.subTest(a8=include_a8, compressed=include_compressed):
                    options = BaseModelTabView._BaseModelTabView__create_dtype_options(
                        None, include_gguf=True, include_a8=include_a8, include_compressed=include_compressed)
                    values = [value for _, value in options]
                    self.assertEqual(len(values), len(set(values)))
                    for dtype, _ in A8_DTYPES:
                        self.assertEqual(dtype in values, include_a8)
                    for dtype in (DataType.INT_W8A8_COMPRESSED, DataType.FLOAT_W8A8_COMPRESSED):
                        self.assertEqual(dtype in values, include_a8 and include_compressed)

    def test_svd_rejects_a8_before_replacing_layers(self):
        config = QuantizationConfig.default_values()
        config.svd_dtype = DataType.BFLOAT_16
        for dtype, _ in A8_DTYPES:
            with self.subTest(dtype=dtype):
                model = nn.Sequential(nn.Linear(64, 64))
                original = model[0]
                with self.assertRaisesRegex(ValueError, "SVDQuant cannot be used with bfloat16 A8"):
                    replace_linear_with_quantized_layers(model, dtype, [], config, copy_parameters=True)
                self.assertIs(model[0], original)

    def test_svd_still_accepts_weight_quantization(self):
        config = QuantizationConfig.default_values()
        config.svd_dtype = DataType.BFLOAT_16
        for dtype in (DataType.INT_W8A8, DataType.FLOAT_W8A8):
            with self.subTest(dtype=dtype):
                model = nn.Sequential(nn.Linear(64, 64))
                replace_linear_with_quantized_layers(model, dtype, [], config, copy_parameters=True)
                self.assertIsInstance(model[0], BaseLinearSVD)

    def test_stream_initializes_a8_compute_dtype_on_each_materialization(self):
        # Start from fp32 checkpoint weights and exercise the real streamed loader, including eviction/reload.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "weights.safetensors")
            weights = {"0.weight": torch.randn(64, 64), "0.bias": torch.randn(64)}
            save_file(weights, path)
            for dtype, activation_dtype in A8_DTYPES:
                for quantize in (False, True):
                    with self.subTest(dtype=dtype, quantize=quantize):
                        model = nn.Sequential(LinearA8(activation_dtype, 64, 64, device="meta"))
                        for train_dtype in (DataType.BFLOAT_16, DataType.FLOAT_32):
                            stream_module_from_checkpoint(
                                model, torch.device("cpu"), dict.fromkeys(weights, path),
                                dtype, train_dtype, [], None, quantize,
                            )
                            self.assertEqual(model[0].compute_dtype, train_dtype.torch_dtype())
                            self.assertEqual(model[0].weight.dtype, torch.bfloat16)
                            torch.testing.assert_close(model[0].weight, weights["0.weight"].to(torch.bfloat16))
                            self.assertFalse(is_quantized_parameter(model[0], "weight"))
                            evict_to_meta(model)
                            self.assertTrue(model[0].weight.is_meta)

    def test_stream_initializes_a8_with_tied_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "weights.safetensors")
            weights = {"0.weight": torch.randn(64, 64)}
            save_file(weights, path)
            for dtype, activation_dtype in A8_DTYPES:
                with self.subTest(dtype=dtype):
                    model = nn.Sequential(
                        LinearA8(activation_dtype, 64, 64, bias=False, device="meta"),
                        LinearA8(activation_dtype, 64, 64, bias=False, device="meta"),
                    )
                    stream_module_from_checkpoint(
                        model, torch.device("cpu"), dict.fromkeys(weights, path),
                        dtype, DataType.BFLOAT_16, [], {"1.weight": "0.weight"}, True,
                    )
                    self.assertEqual(model[1].compute_dtype, torch.bfloat16)
                    torch.testing.assert_close(model[0].weight, model[1].weight)
                    self.assertNotEqual(model[0].weight.data_ptr(), model[1].weight.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for A8 kernels")
    def test_cuda_resident_forward_backward(self):
        for dtype, _ in A8_DTYPES:
            with self.subTest(dtype=dtype):
                model = nn.Sequential(nn.Linear(64, 64, dtype=torch.bfloat16, device="cuda"))
                replace_linear_with_quantized_layers(model, dtype, [], copy_parameters=True)
                quantize_layers(model, torch.device("cuda"), DataType.BFLOAT_16, None)
                model.requires_grad_(True)
                for rows in (16, 32):
                    x = torch.randn(2, rows // 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    y = model(x)
                    y.float().square().mean().backward()
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.dtype, torch.bfloat16)
                    for tensor in (y, x.grad, model[0].weight.grad, model[0].bias.grad):
                        self.assertTrue(torch.isfinite(tensor).all().item())
                    model.zero_grad(set_to_none=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for A8 kernels")
    def test_cuda_streamed_output_matches_resident(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "weights.safetensors")
            weights = {"0.weight": torch.randn(64, 64), "0.bias": torch.randn(64)}
            save_file(weights, path)
            for dtype, activation_dtype in A8_DTYPES:
                with self.subTest(dtype=dtype):
                    resident = nn.Sequential(LinearA8(activation_dtype, 64, 64)).to(device="cuda", dtype=torch.bfloat16)
                    resident.load_state_dict(weights)
                    quantize_layers(resident, torch.device("cuda"), DataType.BFLOAT_16, None)
                    streamed = nn.Sequential(LinearA8(activation_dtype, 64, 64, device="meta"))
                    stream_module_from_checkpoint(
                        streamed, torch.device("cuda"), dict.fromkeys(weights, path),
                        dtype, DataType.BFLOAT_16, [], None, True,
                    )
                    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
                    output = streamed(x)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    torch.testing.assert_close(output, resident(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
