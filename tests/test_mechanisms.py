"""Check illustrative mathematics and tool-loop failure boundaries."""
import unittest
from examples.s01_grpo import advantages, clipped_surrogate
from examples.s02_moe import route, toy_moe
from examples.s03_mla_cache import cache_sizes
from examples.s04_evaluation import majority_answer, pass_at_k
from examples.s05_harness import dispatch, run


class MechanismTests(unittest.TestCase):
    def test_group_advantages(self):
        for actual, expected in zip(advantages([1, 0, 1, 0]), [1, -1, 1, -1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(advantages([0, 0.5, 2, 3])), 0)

    def test_zero_variance(self):
        self.assertEqual(advantages([0, 0]), [0, 0])
        self.assertEqual(advantages([1, 1]), [0, 0])

    def test_advantage_invalid(self):
        for rewards in ([], [1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                advantages(rewards)

    def test_clipping_sign_matters(self):
        self.assertAlmostEqual(clipped_surrogate(1.5, 1), 1.2)
        self.assertAlmostEqual(clipped_surrogate(1.5, -1), -1.5)
        self.assertAlmostEqual(clipped_surrogate(0.5, -1), -0.8)

    def test_route_bias_does_not_change_original_weights(self):
        ids, weights = route([0.6, 0.3, 0.2, 0.1], [0, 0, 0.5, 0], 2)
        self.assertEqual(ids, [2, 0])
        self.assertAlmostEqual(weights[0], 0.25)
        self.assertAlmostEqual(weights[1], 0.75)
        self.assertAlmostEqual(toy_moe(2, [0.6, 0.3, 0.2, 0.1],
                                     [0, 0, 0.5, 0], 2)[2], 1.2)

    def test_zero_route_mass(self):
        with self.assertRaises(ValueError):
            route([0, 0], [0, 0], 1)

    def test_cache_dimensions_and_scaling(self):
        naive, mla = cache_sizes(tokens=1, layers=1, batch=1)
        self.assertEqual((naive, mla), (81920, 1152))
        self.assertEqual(cache_sizes(tokens=2, layers=1, batch=1),
                         (2*naive, 2*mla))
        with self.assertRaises(ValueError):
            cache_sizes(tokens=0)

    def test_pass_at_k_boundaries(self):
        self.assertAlmostEqual(pass_at_k(5, 2, 1), 0.4)
        self.assertAlmostEqual(pass_at_k(5, 2, 2), 0.7)
        self.assertEqual(pass_at_k(5, 2, 5), 1)
        self.assertEqual(pass_at_k(5, 0, 5), 0)
        with self.assertRaises(ValueError):
            pass_at_k(5, 2, 6)

    def test_majority_can_be_wrong_or_tied(self):
        self.assertEqual(majority_answer([41, 41, 41, 42, 42]), 41)
        self.assertIsNone(majority_answer([41, 42]))

    def test_harness_observation_and_stop(self):
        answer, messages = run()
        self.assertEqual(answer, "42")
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[2]["content"], 42)

    def test_harness_rejects_unknown_and_invalid_tools(self):
        bad_calls = [
            {"name": "shell", "arguments": {}},
            {"name": "add", "arguments": {"a": True, "b": 2}},
            {"name": "add", "arguments": {"a": 1}},
            {"name": "add", "arguments": {"a": 10**7, "b": 2}},
        ]
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                dispatch(call)

    def test_harness_budget(self):
        def never_finishes(_):
            return {"type": "tool_call", "name": "add",
                    "arguments": {"a": 1, "b": 1}}
        with self.assertRaises(RuntimeError):
            run(never_finishes, max_steps=2)


if __name__ == "__main__":
    unittest.main()
