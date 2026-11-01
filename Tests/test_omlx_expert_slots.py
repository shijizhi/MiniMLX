import unittest

from minimlx_runtime.omlx_expert_slots import _use_legacy_prefill


class ExpertSlotsTests(unittest.TestCase):
    def test_large_prefill_preserves_stable_expert_path(self):
        self.assertTrue(_use_legacy_prefill((1, 128, 2560)))
        self.assertFalse(_use_legacy_prefill((1, 8, 2560)))
        self.assertFalse(_use_legacy_prefill((1, 1, 2560)))


if __name__ == "__main__":
    unittest.main()
