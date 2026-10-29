import json
import tempfile
import unittest
from pathlib import Path

from minimlx_runtime.deepseek41_offload import (
    DeepSeek41CacheBudget,
    DeepSeek41TensorLayout,
    EngramPageStore,
)
from minimlx_runtime.offload import Qwen38OffloadSession, build_safetensors_offload_index


class DeepSeek41OffloadTests(unittest.TestCase):
    def test_layout_separates_deepseek41_storage_families(self) -> None:
        layout = DeepSeek41TensorLayout.from_tensor_names(
            [
                "model.encoder.layers.0.weight",
                "model.decoder.layers.0.attn.weight",
                "model.decoder.layers.0.experts.0.up_proj.weight",
                "model.engram.table.weight",
                "model.attention.indexer.weight",
                "model.hyper_connections.combine.weight",
                "model.dspark.draft.weight",
            ]
        )

        self.assertEqual(len(layout.encoder), 1)
        self.assertEqual(len(layout.decoder), 1)
        self.assertEqual(len(layout.routed_experts), 1)
        self.assertEqual(len(layout.engram), 1)
        self.assertEqual(len(layout.sparse_indexer), 1)
        self.assertEqual(len(layout.hyper_connections), 1)
        self.assertEqual(len(layout.dspark), 1)

    def test_cache_budget_keeps_engram_separate(self) -> None:
        budget = DeepSeek41CacheBudget(
            resident_bytes=100,
            encoder_bytes=20,
            decoder_kv_bytes=30,
            engram_page_bytes=50,
        )
        self.assertEqual(budget.engram_page_bytes, 50)
        with self.assertRaises(ValueError):
            DeepSeek41CacheBudget(10, 5, 5, 1)

    def test_engram_store_reads_one_page_without_loading_full_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = json.dumps(
                {
                    "model.engram.table.weight": {
                        "dtype": "U8",
                        "shape": [6, 2],
                        "data_offsets": [0, 12],
                    }
                },
                separators=(",", ":"),
            ).encode()
            (root / "model-00001.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + bytes(range(12))
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.engram.table.weight": "model-00001.safetensors"
                        }
                    }
                )
            )
            index = build_safetensors_offload_index(root)
            session = Qwen38OffloadSession.open(root, index_path=index)
            store = EngramPageStore(
                session, ("model.engram.table.weight",), page_rows=2
            )

            self.assertEqual(
                store.read_page("model.engram.table.weight", 1).tolist(),
                [[4, 5], [6, 7]],
            )


if __name__ == "__main__":
    unittest.main()
