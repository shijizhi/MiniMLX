import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from minimlx_runtime.ple_offload import PLE_MARKER, prepare_paged_ple_view, init_shared_ple_ranges, supports_paged_ple


class PagedPLETests(unittest.TestCase):
    def test_detects_supported_and_mixed_ple_layouts(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            paged = source / ".minimlx-ssd"
            paged.mkdir()
            names = ["model" + PLE_MARKER + str(i) for i in range(2)]
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name + ".weight": "weights" for name in names}}))
            quant = {"bits": 4, "group_size": 32, "mode": "affine"}
            config = {"model_type": "qwen4_exp", "quantization": quant}
            (source / "config.json").write_text(json.dumps(config))
            self.assertTrue(supports_paged_ple(paged))
            quant[names[1]] = {"bits": 4, "group_size": 16, "mode": "nvfp4"}
            (source / "config.json").write_text(json.dumps(config))
            self.assertFalse(supports_paged_ple(paged))

    def test_ranges_share_one_mapping_and_preserve_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "weights.safetensors").write_bytes(bytes(range(80)))
            store = types.SimpleNamespace(_tensor_layout=(("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")),
                                          _tensor_names=("weight", "scales", "biases"), row_width=8, _group_size=4)
            def shard(start, base):
                return {"row_start": start, "row_count": 1,
                        "weight": {"dtype": "U32", "file": "weights.safetensors", "offset": base, "shape": [1, 1]},
                        "scales": {"dtype": "BF16", "file": "weights.safetensors", "offset": base + 4, "shape": [1, 2]},
                        "biases": {"dtype": "BF16", "file": "weights.safetensors", "offset": base + 8, "shape": [1, 2]}}
            init_shared_ple_ranges(store, {"shards": [shard(0, 0), shard(1, 12)]}, root)
            self.assertEqual(len(store._minimlx_shared_mmaps), 1)
            self.assertEqual(store.row_count, 2)
            self.assertEqual(store._shards[1][2]["weight"].view("u1").tolist(), [[12, 13, 14, 15]])
            self.assertFalse(store._shards[0][2]["weight"].flags.writeable)

    def test_view_links_weights_and_keeps_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            paged = source / ".minimlx-ssd"
            (paged / "experts").mkdir(parents=True)
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model" + PLE_MARKER + "0.weight": "source.safetensors"}}))
            config = {"model_type": "qwen4_exp", "text_config": {}, "quantization": {"bits": 4, "model" + PLE_MARKER + "0": {"bits": 4}}}
            (paged / "config.json").write_text(json.dumps(config))
            (paged / "offload_index.json").write_text('{}')
            (paged / "tokenizer.json").write_text('{}')
            (paged / "resident-0001.safetensors").write_bytes(b"weights")
            (paged / "experts/layer_0000.safetensors").write_bytes(b"experts")
            module = types.ModuleType("mlx_vlm.models.qwen4_exp.ple_storage")
            def build(original, output, cache_rows):
                self.assertEqual(original, source.resolve())
                self.assertEqual(cache_rows, 8192)
                output.write_text(json.dumps({"source_root": os.path.relpath(original, output.parent)}))
            module.build_quantized_ple_manifest = build
            with patch.dict(sys.modules, {"mlx_vlm.models.qwen4_exp.ple_storage": module}):
                view = prepare_paged_ple_view(paged)
                self.assertEqual(prepare_paged_ple_view(paged), view)
            self.assertEqual(os.stat(paged / "resident-0001.safetensors").st_ino, os.stat(view / "resident-0001.safetensors").st_ino)
            self.assertEqual(os.stat(paged / "experts/layer_0000.safetensors").st_ino, os.stat(view / "experts/layer_0000.safetensors").st_ino)
            self.assertEqual(json.loads((paged / "config.json").read_text()), config)
            generated = json.loads((view / "config.json").read_text())
            self.assertEqual(generated["text_config"]["ple_storage"]["manifest"], "ple-store.json")
            self.assertFalse(any(PLE_MARKER in key for key in generated["quantization"]))
            manifest = json.loads((view / "ple-store.json").read_text())
            self.assertEqual((view / manifest["source_root"]).resolve(), source.resolve())


if __name__ == "__main__":
    unittest.main()
