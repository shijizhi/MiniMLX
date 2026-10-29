import unittest

from minimlx_runtime.deepseek41_executor import (
    DeepSeek41CEDCache,
    DeepSeek41EngramLookup,
    DeepSeek41Router,
)


class DeepSeek41ExecutorTests(unittest.TestCase):
    def test_router_selects_top_k_and_normalizes_scores(self) -> None:
        router = DeepSeek41Router(num_experts=4, top_k=2)
        result = router.route([[0.1, 0.9, 0.2, 0.4]])

        self.assertEqual(result.expert_ids.tolist(), [[1, 3]])
        self.assertAlmostEqual(float(result.scores.sum()), 1.0, places=6)

    def test_image_tokens_use_a_separate_routing_bias(self) -> None:
        router = DeepSeek41Router(num_experts=3, top_k=1)
        result = router.route(
            [[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]],
            image_token_mask=[False, True],
            image_bias=[0.0, 0.0, 1.0],
        )

        self.assertEqual(result.expert_ids.tolist(), [[0], [2]])

    def test_ced_cache_requires_encoder_before_decoder(self) -> None:
        cache = DeepSeek41CEDCache()
        with self.assertRaises(RuntimeError):
            cache.append_decoder_state(object())
        cache.set_encoder_output(object())
        cache.append_decoder_state(object())
        self.assertEqual(cache.decoder_length, 1)
        cache.clear()
        self.assertEqual(cache.decoder_length, 0)

    def test_engram_lookup_emits_rows_for_configured_ngram_orders(self) -> None:
        seen = []
        lookup = DeepSeek41EngramLookup(
            lambda rows: seen.extend(rows) or rows,
            hash_fn=lambda tokens: sum(tokens),
            orders=(2, 3),
        )

        self.assertEqual(lookup.lookup([1, 2, 3]), [3, 5, 6])
        self.assertEqual(seen, [3, 5, 6])


if __name__ == "__main__":
    unittest.main()
