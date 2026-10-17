import copy
import importlib
import unittest
from unittest.mock import patch

from modules.model.BaseModel import BaseModel
from modules.util.checkpointing_util import create_checkpoint
from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor, SyncEvent

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32).requires_grad_(False)
        self.gain = None

    def forward(self, hidden_states):
        output = self.linear(hidden_states).tanh()
        if self.gain is not None:
            output = output * self.gain
        return hidden_states + output


class Network(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(6)])
        if trainable:
            # Like LoRA, adapters are owned separately from the offloaded base layers.
            self.gains = nn.ParameterList([nn.Parameter(torch.ones(32)) for _ in self.layers])
            for layer, gain in zip(self.layers, self.gains, strict=True):
                object.__setattr__(layer, "gain", gain)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@unittest.skipUnless(torch.cuda.is_available(), "CUDA integration tests")
class OffloadMergeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1649)
        self.config = TrainConfig.default_values()
        self.config.train_device = "cuda:0"
        self.config.temp_device = "cpu"
        self.config.async_offloading = True
        self.part = TrainModelPartConfig.default_values()
        self.part.offload_fraction = 0.75
        self.part.activation_offloading = True

    def make_model(self, *, streaming=False, simplex=False, compiled=False, input_grad=True):
        network = Network(trainable=not input_grad)
        reference = copy.deepcopy(network).cuda()
        state = {name: tensor.clone() for name, tensor in network.state_dict().items()}
        network = network.to("meta" if streaming else "cuda:0")
        conductor = LayerOffloadConductor(network, self.config, self.part, simplex=simplex)
        stream_calls = []

        def materialize(module, device, dtype, prefix="", prepare_only=False, **kwargs):
            if prepare_only:
                return
            stream_calls.append(prefix)
            for module_name, child in module.named_modules():
                for name, parameter in list(child.named_parameters(recurse=False)):
                    if parameter.is_meta:
                        key = ".".join(filter(None, (prefix, module_name, name))).replace(".checkpoint", "")
                        child.register_parameter(name, nn.Parameter(
                            state[key].to(device), requires_grad=parameter.requires_grad))

        for index, layer in enumerate(network.layers):
            network.layers[index] = create_checkpoint(
                layer, ["hidden_states"], conductor, layer_index=index, compile=compiled)
        return network, reference, conductor, materialize if streaming else None, stream_calls

    def check_pass(self, network, reference, *, input_grad=True):
        network.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        data = torch.randn(4, 8, 32, device="cuda:0")
        actual_input = data.clone().requires_grad_(input_grad)
        expected_input = data.clone().requires_grad_(input_grad)
        expected = reference(expected_input)
        actual = network(actual_input)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        if input_grad:
            torch.testing.assert_close(actual_input.grad, expected_input.grad)
        else:
            actual_gradients = [p.grad for p in network.parameters() if p.requires_grad]
            expected_gradients = [p.grad for p in reference.parameters() if p.requires_grad]
            self.assertEqual(len(actual_gradients), len(expected_gradients))
            for actual_grad, expected_grad in zip(actual_gradients, expected_gradients, strict=True):
                self.assertIsNotNone(actual_grad)
                torch.testing.assert_close(actual_grad, expected_grad)
        torch.cuda.synchronize()

    def check_cycle(self, *, streaming=False, simplex=False, cache_in_ram=True, compiled=False, input_grad=True):
        network, reference, conductor, materialize, calls = self.make_model(
            streaming=streaming, simplex=simplex, compiled=compiled, input_grad=input_grad)
        kwargs = {"train_dtype": DataType.FLOAT_32, "name": "test", "materialize_fn": materialize,
                  "cache_in_ram": cache_in_ram}
        try:
            self.assertTrue(conductor.materialize(**kwargs))
            self.assertFalse(conductor.materialize(**kwargs))
            initial_calls = len(calls)
            syncs = []
            original_sync = SyncEvent.synchronize

            def record_sync(event, message=None):
                syncs.append(message)
                return original_sync(event, message)

            with patch.object(SyncEvent, "synchronize", record_sync):
                for _ in range(2):
                    self.check_pass(network, reference, input_grad=input_grad)
            if self.part.activation_offloading and self.config.async_offloading:
                self.assertTrue(any("compute" in message for message in syncs))
                self.assertTrue(any("transfer" in message for message in syncs))
                self.assertGreater(conductor._LayerOffloadConductor__temp_device_activations_allocator.allocated_bytes, 0)
            with torch.no_grad():
                data = torch.randn(4, 8, 32, device="cuda:0")
                torch.testing.assert_close(network(data), reference(data))
            to_meta = streaming and not cache_in_ram
            self.assertTrue(conductor.evict(to_meta=to_meta))
            self.assertFalse(conductor.evict(to_meta=to_meta))
            self.assertTrue(all(p.device.type == ("meta" if to_meta else "cpu") for p in network.parameters()))
            self.assertTrue(conductor.materialize(**kwargs))
            if streaming:
                self.assertEqual(len(calls) > initial_calls, not cache_in_ram)
            self.check_pass(network, reference, input_grad=input_grad)
        finally:
            conductor.evict(to_meta=streaming and not cache_in_ram)

    def test_resident_async_gradients_and_no_grad_input(self):
        for input_grad in (True, False):
            with self.subTest(input_grad=input_grad):
                self.check_cycle(input_grad=input_grad)

    def test_resident_synchronous_gradients(self):
        self.config.async_offloading = False
        self.check_cycle()

    def test_weight_only_offload(self):
        self.part.activation_offloading = False
        self.check_cycle()

    def test_activation_only_offload(self):
        self.part.offload_fraction = 0
        self.check_cycle()

    def test_streaming_static_and_simplex_cycles(self):
        for simplex in (False, True):
            for cache_in_ram in (False, True):
                with self.subTest(simplex=simplex, cache_in_ram=cache_in_ram):
                    self.check_cycle(streaming=True, simplex=simplex, cache_in_ram=cache_in_ram)

    def test_compiled_autograd_static_and_simplex(self):
        # AOTAutograd exercises compiled checkpoint forward/backward without requiring a C++/Triton toolchain.
        original_compile = torch.compile
        with patch.object(torch, "compile", lambda *args, **kwargs: original_compile(
                *args, **kwargs, backend="aot_eager")):
            self.check_cycle(compiled=True)
            self.check_cycle(streaming=True, simplex=True, cache_in_ram=False, compiled=True)

    def test_activation_layout_and_unmaterialized_guards(self):
        _, _, conductor, _, _ = self.make_model()
        data = torch.randn(8, 16, device="cuda:0")
        self.assertFalse(conductor.evict())
        conductor.start_forward(backward_follows=True)
        conductor.before_layer(0, is_forward=True)
        conductor.after_layer(0, [data])
        conductor.prefetch_activations(0)
        self.assertIs(conductor.pack_activation(0, data), data)
        try:
            conductor.materialize()
            for tensor in (data, data.t(), data[:, ::2]):
                with self.subTest(stride=tensor.stride()):
                    conductor.start_forward(backward_follows=True)
                    handle = conductor.pack_activation(0, tensor)
                    self.assertIsNot(handle, tensor)
                    conductor.prefetch_activations(0)
                    restored = conductor.unpack_activation(handle)
                    self.assertEqual(restored.stride(), tensor.stride())
                    torch.testing.assert_close(restored, tensor)
            self.assertIs(conductor.pack_activation(5, data), data)
        finally:
            conductor.evict()

    def test_failed_stream_rolls_back_and_retries(self):
        network, reference, conductor, materialize, _ = self.make_model(streaming=True, simplex=True)

        def fail(module, device, dtype, prefix="", **kwargs):
            if prefix == "layers.1":
                raise RuntimeError("simulated stream failure")
            materialize(module, device, dtype, prefix, **kwargs)

        try:
            with self.assertRaisesRegex(RuntimeError, "simulated stream failure"):
                conductor.materialize(materialize_fn=fail, cache_in_ram=False)
            self.assertTrue(all(p.is_meta for p in network.parameters()))
            self.assertFalse(conductor.evict(to_meta=True))
            self.assertTrue(conductor.materialize(materialize_fn=materialize, cache_in_ram=False))
            self.check_pass(network, reference)
        finally:
            conductor.evict(to_meta=True)

    def test_base_model_conductor_dispatch(self):
        network, _, conductor, materialize, _ = self.make_model(streaming=True, simplex=True)
        model = BaseModel(ModelType.FLUX_DEV_1)
        model.train_config = self.config
        model.transformer = network
        model.offload_conductor["transformer"] = conductor
        model.materialize_fn["transformer"] = materialize
        model.cache_in_ram["transformer"] = False
        base_module = importlib.import_module("modules.model.BaseModel")
        try:
            model.materialize("transformer")
            self.assertTrue(any(p.is_cuda for p in network.parameters()))
            model.evict("transformer")
            self.assertTrue(all(p.is_meta for p in network.parameters()))
            with patch.object(base_module, "torch_gc") as collect:
                model.evict("transformer")
                collect.assert_not_called()
        finally:
            conductor.evict(to_meta=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
