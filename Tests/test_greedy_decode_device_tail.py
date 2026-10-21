import unittest
from types import ModuleType
from unittest.mock import patch

# Import the adapter with a tiny fake MLX module. These tests exercise a pure
# Python decode-tail helper and must remain runnable without Metal access.
_mlx = ModuleType("mlx")
_mlx_core = ModuleType("mlx.core")
_mlx.core = _mlx_core
with patch.dict("sys.modules", {"mlx": _mlx, "mlx.core": _mlx_core}):
    from minimlx_runtime.adapters.qwen35 import Qwen35Adapter


class FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def astype(self, _dtype):
        return self

    def __getitem__(self, key):
        # The production path adds a singleton sequence axis, then selects
        # the final sequence position from model logits.
        if key == (slice(None), None):
            return self
        return self

    def tolist(self):
        return list(self.values)


class FakeModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, tokens, *, cache):
        self.inputs.append(tokens)
        return FakeArray([token + 1 for token in tokens.values])


class FakeMx:
    uint32 = object()

    @staticmethod
    def argmax(logits, *, axis):
        assert axis == -1
        return logits

    @staticmethod
    def async_eval(_value):
        pass

    @staticmethod
    def concatenate(values, *, axis):
        assert axis == 0
        return FakeArray(token for value in values for token in value.values)

    @staticmethod
    def eval(_value):
        pass


class GreedyDecodeDeviceTailTests(unittest.TestCase):
    def test_tail_preserves_token_arrays_and_collects_once(self):
        adapter = Qwen35Adapter()
        model = FakeModel()
        adapter.model = model
        sampled = FakeArray([4])

        tail = adapter._greedy_decode_device_tail(
            FakeMx, sampled, cache=object(), token_count=3
        )

        self.assertEqual(tail, [5, 6, 7])
        self.assertEqual(len(model.inputs), 3)
        self.assertTrue(all(isinstance(tokens, FakeArray) for tokens in model.inputs))
        self.assertEqual(sampled.values, [4])

    def test_zero_token_tail_does_no_work(self):
        adapter = Qwen35Adapter()
        model = FakeModel()
        adapter.model = model

        self.assertEqual(
            adapter._greedy_decode_device_tail(FakeMx, FakeArray([4]), object(), 0), []
        )
        self.assertEqual(model.inputs, [])
