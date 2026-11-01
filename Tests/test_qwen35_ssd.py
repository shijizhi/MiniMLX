import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from minimlx_runtime.adapters.qwen35 import Qwen35Adapter
from minimlx_runtime.types import Acceleration


class Qwen35PagingTests(unittest.TestCase):
    def test_paged_text_model_uses_vlm_expert_pager(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "offload_index.json").write_text("{}")
            descriptor = SimpleNamespace(model_dir=directory)
            plan = SimpleNamespace(accelerations=set())
            adapter = Qwen35Adapter()
            with patch.object(adapter, "_load_vlm_backend", return_value="paged") as load:
                self.assertEqual(adapter.load(descriptor, plan), "paged")
                load.assert_called_once_with(descriptor, plan)

    def test_paged_model_does_not_silently_discard_a_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "offload_index.json").write_text("{}")
            descriptor = SimpleNamespace(model_dir=directory)
            plan = SimpleNamespace(accelerations={Acceleration.MTP},
                                   acceleration_components={Acceleration.MTP: "/draft"})
            with self.assertRaisesRegex(RuntimeError, "not yet supported"):
                Qwen35Adapter().load(descriptor, plan)


if __name__ == "__main__":
    unittest.main()
