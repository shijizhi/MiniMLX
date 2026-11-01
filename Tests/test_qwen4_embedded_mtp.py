import json
import tempfile
import unittest
from pathlib import Path

from minimlx_runtime.qwen4_mtp import embedded_mtp_source


class EmbeddedMTPTests(unittest.TestCase):
    def test_requires_configuration_and_actual_mtp_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            config = {"model_type": "qwen4_exp", "text_config": {"mtp_num_hidden_layers": 1}}
            (source / "config.json").write_text(json.dumps(config))
            index = source / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"model.layers.0.weight": "a"}}))
            self.assertIsNone(embedded_mtp_source(source))
            index.write_text(json.dumps({"weight_map": {"mtp.fc_hidden.weight": "a"}}))
            self.assertEqual(embedded_mtp_source(source), source)
            self.assertEqual(embedded_mtp_source(source / ".minimlx-ssd"), source)
            config["text_config"]["mtp_num_hidden_layers"] = 0
            (source / "config.json").write_text(json.dumps(config))
            self.assertIsNone(embedded_mtp_source(source))


if __name__ == "__main__":
    unittest.main()
