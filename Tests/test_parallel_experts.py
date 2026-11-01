import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from minimlx_runtime.parallel_experts import ParallelExpertReads, exact_expert_rows, install_parallel_expert_reads


class ParallelReadTests(unittest.TestCase):
    def test_exact_rows_option_applies_without_verify_scope(self):
        import numpy as np
        calls = []
        class OffloadedSwitchGLU:
            def _proj(self, x, weight, scales, biases, quant):
                calls.append(x.shape[0])
                return x
        switch = OffloadedSwitchGLU()
        model = types.SimpleNamespace(moe_offload_store=object(), named_modules=lambda: [("layer", switch)])
        fake = types.ModuleType("mlx_vlm.models.switch_layers")
        fake.OffloadedSwitchGLU = OffloadedSwitchGLU
        mx = types.ModuleType("mlx.core")
        mx.concatenate = np.concatenate
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx_vlm.models.switch_layers": fake, "mlx": mlx, "mlx.core": mx}):
            controller = install_parallel_expert_reads(model, exact_rows=True)
            try:
                switch._proj(np.ones((3, 2)), None, None, None, None)
                self.assertEqual(calls, [1, 1, 1])
            finally:
                controller.close()

    def test_exact_expert_projection_splits_rows_only_in_verify_scope(self):
        import numpy as np
        calls = []
        class OffloadedSwitchGLU:
            def _proj(self, x, weight, scales, biases, quant):
                calls.append(x.shape[0])
                return x * 2
        switch = OffloadedSwitchGLU()
        model = types.SimpleNamespace(moe_offload_store=object(), named_modules=lambda: [("layer", switch)])
        fake = types.ModuleType("mlx_vlm.models.switch_layers")
        fake.OffloadedSwitchGLU = OffloadedSwitchGLU
        mx = types.ModuleType("mlx.core")
        mx.concatenate = np.concatenate
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx_vlm.models.switch_layers": fake, "mlx": mlx, "mlx.core": mx}):
            controller = install_parallel_expert_reads(model)
            try:
                x = np.array([[1, 2], [3, 4], [5, 6]])
                with exact_expert_rows():
                    np.testing.assert_array_equal(switch._proj(x, None, None, None, None), x * 2)
                switch._proj(x, None, None, None, None)
                self.assertEqual(calls, [1, 1, 1, 3])
            finally:
                controller.close()

    def test_expert_evicted_during_call_uses_direct_read(self):
        import numpy as np

        store = types.SimpleNamespace(_lru={(15, 2): 1}, num_experts=512)
        store._ensure_loaded = lambda *args: self.fail("shard should not be reopened")
        controller = ParallelExpertReads(store, workers=1)
        controller.fds[15] = os.open(os.devnull, os.O_RDONLY)
        meta = {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
        controller.headers[15] = (0, {
            1: [("e1.gate_proj.weight", meta)],
            2: [("e2.gate_proj.weight", meta)],
        })
        mx = types.ModuleType("mlx.core")
        mx.array, mx.bfloat16 = np.array, object()
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        def read(fd, base, entries):
            return [(entries[0][0], meta, bytearray([42]))]
        def original(layer, x, indices):
            mapping = {}
            store._ensure_loaded(15, 2, mapping)
            store._ensure_loaded(15, 1, mapping)
            return sorted(mapping)
        try:
            with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}), patch.object(controller, "_read", side_effect=read):
                result = controller.run(types.SimpleNamespace(layer_id=15), None, np.array([[1, 2]]), original)
            self.assertEqual(result, ["e1.gate_proj.weight", "e2.gate_proj.weight"])
        finally:
            controller.close()

    def test_large_prefill_reads_experts_without_reopening_the_shard(self):
        import numpy as np

        store = types.SimpleNamespace(_lru={}, num_experts=512)
        def unexpected_load(*args):
            raise AssertionError("should not reopen the safetensors shard")
        store._ensure_loaded = unexpected_load
        controller = ParallelExpertReads(store, workers=1)
        controller.fds[15] = os.open(os.devnull, os.O_RDONLY)
        meta = {"dtype": "U8", "shape": [1], "data_offsets": [0, 129 * 1024 * 1024]}
        controller.headers[15] = (0, {1: [("e1.gate_proj.weight", meta)]})
        mx = types.ModuleType("mlx.core")
        mx.array = np.array
        mx.bfloat16 = object()
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        calls = []
        def read(fd, base, entries):
            calls.append(1)
            return [("e1.gate_proj.weight", {"dtype": "U8", "shape": [1]}, bytearray([42]))]
        def original(layer, x, indices):
            mapping = {}
            store._ensure_loaded(15, 1, mapping)
            self.assertEqual(int(mapping["e1.gate_proj.weight"][0]), 42)
            return "ok"
        try:
            with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}), patch.object(controller, "_read", side_effect=read):
                self.assertEqual(controller.run(types.SimpleNamespace(layer_id=15), None, np.array([[1]]), original), "ok")
            self.assertEqual(len(calls), 1)
            self.assertIs(store._ensure_loaded, unexpected_load)
        finally:
            controller.close()

    def test_read_handles_short_chunks_and_preserves_offsets(self):
        with tempfile.TemporaryFile() as file:
            file.write(b"header0123456789")
            file.flush()
            original = os.preadv
            with patch("minimlx_runtime.parallel_experts.os.preadv", side_effect=lambda fd, buffers, off: original(fd, [buffers[0][:2]], off)):
                result = ParallelExpertReads._read(file.fileno(), 6, [("weight", {"data_offsets": [2, 7]})])
            self.assertEqual(bytes(result[0][2]), b"23456")

    def test_truncated_read_fails(self):
        with tempfile.TemporaryFile() as file:
            file.write(b"abc")
            file.flush()
            with self.assertRaises(OSError):
                ParallelExpertReads._read(file.fileno(), 0, [("weight", {"data_offsets": [0, 4]})])


if __name__ == "__main__":
    unittest.main()
