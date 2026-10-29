import json
import tempfile
import unittest
from pathlib import Path

from minimlx_runtime.offload import (
    ExpertCache,
    GIB,
    MmapWeightStore,
    PrefetchScheduler,
    ResidentBudget,
    Qwen38OffloadSession,
    build_safetensors_offload_index,
    make_qwen38_offload_plan,
    qwen38_should_use_ssd_offload,
)


class OffloadPlanningTests(unittest.TestCase):
    def test_auto_mode_keeps_qwen38_resident_when_safe(self) -> None:
        self.assertFalse(
            qwen38_should_use_ssd_offload(
                4 * GIB,
                total_memory_bytes=32 * GIB,
                available_memory_bytes=20 * GIB,
            )
        )

    def test_auto_mode_switches_to_ssd_when_peak_load_is_unsafe(self) -> None:
        self.assertTrue(
            qwen38_should_use_ssd_offload(
                20 * GIB,
                total_memory_bytes=32 * GIB,
                available_memory_bytes=20 * GIB,
            )
        )

    def test_expert_cache_is_byte_bounded_lru(self) -> None:
        cache = ExpertCache(capacity_bytes=5)
        cache.put("a", b"a", 2)
        cache.put("b", b"b", 2)
        self.assertEqual(cache.get("a"), b"a")
        cache.put("c", b"c", 3)

        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), b"a")
        self.assertEqual(cache.get("c"), b"c")
        self.assertEqual(cache.used_bytes, 5)

    def test_prefetch_scheduler_deduplicates_and_batches(self) -> None:
        scheduler = PrefetchScheduler(max_batch_size=2)
        scheduler.request(["expert.1", "expert.2", "expert.1", "ple.0"])

        self.assertEqual(scheduler.pop_batch(), ("expert.1", "expert.2"))
        self.assertEqual(scheduler.pop_batch(), ("ple.0",))
        self.assertEqual(scheduler.pop_batch(), ())

    def test_safe_budget_applies_reserves_and_configured_cap(self) -> None:
        budget = ResidentBudget(
            total_bytes=24 * GIB,
            available_bytes=20 * GIB,
            os_reserve_bytes=2 * GIB,
            runtime_reserve_bytes=1 * GIB,
            configured_bytes=10 * GIB,
        )

        self.assertEqual(budget.safe_bytes, 10 * GIB)

    def test_offload_plan_can_start_when_full_model_does_not_fit(self) -> None:
        budget = ResidentBudget(
            total_bytes=32 * GIB,
            available_bytes=20 * GIB,
            os_reserve_bytes=2 * GIB,
            runtime_reserve_bytes=2 * GIB,
        )
        plan = make_qwen38_offload_plan(
            "/models/qwen3.8-flash-next",
            total_weight_bytes=24 * GIB,
            dense_resident_bytes=8 * GIB,
            expert_bytes=14 * GIB,
            ple_bytes=2 * GIB,
            budget=budget,
        )

        self.assertTrue(plan.can_start)
        self.assertFalse(plan.fits_without_offload)
        self.assertEqual(plan.stream_bytes, 16 * GIB)
        self.assertIn("offload required", plan.reason)

    def test_without_offload_requires_full_weight_set(self) -> None:
        budget = ResidentBudget(
            total_bytes=16 * GIB,
            available_bytes=16 * GIB,
            os_reserve_bytes=1 * GIB,
            runtime_reserve_bytes=1 * GIB,
        )
        plan = make_qwen38_offload_plan(
            "/models/qwen3.8-flash-next",
            total_weight_bytes=20 * GIB,
            dense_resident_bytes=8 * GIB,
            expert_bytes=10 * GIB,
            ple_bytes=2 * GIB,
            budget=budget,
            use_ssd_offload=False,
        )

        self.assertFalse(plan.can_start)
        self.assertEqual(plan.stream_bytes, 0)
        self.assertEqual(plan.minimum_resident_bytes, 20 * GIB)

    def test_indexed_store_reads_only_requested_tensor_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "weights-00001.bin"
            shard.write_bytes(b"prefix-abcdef-suffix")
            index = root / "offload-index.json"
            index.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "layer.weight": {
                                "file": "weights-00001.bin",
                                "offset": 7,
                                "length": 6,
                            }
                        }
                    }
                )
            )

            store = MmapWeightStore.from_index(root, index)

            self.assertEqual(store.read("layer.weight"), b"abcdef")
            self.assertEqual(store.read("layer.weight", offset=1, length=3), b"bcd")
            with self.assertRaises(ValueError):
                store.read("layer.weight", offset=5, length=2)

    def test_index_rejects_absolute_shard_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "offload-index.json"
            index.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "layer.weight": {
                                "file": str(root / "weights.bin"),
                                "offset": 0,
                                "length": 1,
                            }
                        }
                    }
                )
            )

            with self.assertRaises(ValueError):
                MmapWeightStore.from_index(root, index)

    def test_builds_mmap_index_from_safetensors_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = json.dumps(
                {"layer.weight": {"dtype": "U8", "shape": [6], "data_offsets": [0, 6]}},
                separators=(",", ":"),
            ).encode()
            (root / "model-00001.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"abcdef"
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.weight": "model-00001.safetensors",
                        }
                    }
                )
            )

            index_path = build_safetensors_offload_index(root)
            store = MmapWeightStore.from_index(root, index_path)

            self.assertEqual(store.read("layer.weight"), b"abcdef")
            session = Qwen38OffloadSession.open(root, index_path=index_path)
            self.assertEqual(session.read_array("layer.weight").tolist(), [97, 98, 99, 100, 101, 102])

    def test_qwen38_session_builds_index_and_prefetches_cached_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = json.dumps(
                {"layer.weight": {"dtype": "U8", "shape": [6], "data_offsets": [0, 6]}},
                separators=(",", ":"),
            ).encode()
            (root / "model-00001.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"abcdef"
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": "model-00001.safetensors"}})
            )

            session = Qwen38OffloadSession.open(root, cache_bytes=32)
            session.request_prefetch(["layer.weight"])
            self.assertEqual(session.prefetch_next_batch()["layer.weight"], b"abcdef")
            self.assertEqual(session.read("layer.weight"), b"abcdef")
            self.assertEqual(session.cache.hits, 1)


if __name__ == "__main__":
    unittest.main()
