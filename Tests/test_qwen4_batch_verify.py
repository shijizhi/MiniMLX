import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from minimlx_runtime.qwen4_batch_verify import install_batch_mtp_verifier, _use_omlx_row_exact


class BatchVerifierTransactionTests(unittest.TestCase):
    def test_omlx_kernel_requires_its_verified_mlx_version(self):
        self.assertFalse(_use_omlx_row_exact("0.32.2", "auto"))
        self.assertTrue(_use_omlx_row_exact("0.32.4", "auto"))
        self.assertFalse(_use_omlx_row_exact("0.32.4", "0"))
        with self.assertRaisesRegex(RuntimeError, "0.32.4"):
            _use_omlx_row_exact("0.32.2", "1")

    def test_failure_aborts_and_success_returns_transaction(self):
        events = []

        class Transaction:
            def abort(self):
                events.append("abort")

        class Base:
            def __call__(self, lm, inputs, **kwargs):
                events.append(("positions", kwargs["position_ids"].tolist()))
                if lm.fail:
                    raise RuntimeError("failure")
                return types.SimpleNamespace(hidden_states=[inputs])

        mx = types.ModuleType("mlx.core")
        mx.array, mx.int64 = np.ndarray, np.int64
        mx.full, mx.arange, mx.broadcast_to = np.full, np.arange, np.broadcast_to
        mx.concatenate = np.concatenate
        mx.async_eval = lambda *args: None
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        nn = types.ModuleType("mlx.nn")
        nn.QuantizedLinear = type("QuantizedLinear", (), {})
        mlx.nn = nn
        layer = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
        layer.Qwen4ExpBatchInvariantForward = Base
        state = types.ModuleType("mlx_vlm.speculative.cache_state")
        state.start_speculative_cache = lambda cache, length: Transaction()
        modules = {"mlx": mlx, "mlx.core": mx, "mlx.nn": nn,
                   "mlx_vlm.models.qwen4_exp.language": layer,
                   "mlx_vlm.speculative.cache_state": state}
        entry = types.SimpleNamespace(offset=7, state=[])
        lm = types.SimpleNamespace(model=types.SimpleNamespace(fa_idx=0),
                                   _rope_deltas=None, _position_ids=None, fail=False,
                                   speculative_verify_hidden=lambda inputs, cache: "exact")
        model = types.SimpleNamespace(language_model=lm)
        with patch.dict(sys.modules, modules):
            install_batch_mtp_verifier(model)
            calls = []
            def projection(x):
                calls.append(x.shape[1])
                return x * 2
            inputs = np.array([[[1], [2], [3]]])
            np.testing.assert_array_equal(model._minimlx_qwen4_batch_verifier._linear(projection, inputs), inputs * 2)
            self.assertEqual(calls, [1, 1, 1])
            _, _, transaction = lm.speculative_verify_hidden(np.array([[1, 2]]), [entry])
            self.assertIsInstance(transaction, Transaction)
            self.assertIn(("positions", [[7, 8]]), events)
            self.assertEqual(lm.speculative_verify_hidden(np.array([[1], [2]]), [entry]), "exact")
            lm._position_ids = np.array([[[0]]])
            self.assertEqual(lm.speculative_verify_hidden(np.array([[1]]), [entry]), "exact")
            lm._position_ids = None
            lm.fail = True
            with self.assertRaisesRegex(RuntimeError, "failure"):
                lm.speculative_verify_hidden(np.array([[1, 2]]), [entry])
            self.assertEqual(events[-1], "abort")


if __name__ == "__main__":
    unittest.main()
