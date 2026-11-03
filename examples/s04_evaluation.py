"""Synthetic exact-match evaluation, not a DeepSeek benchmark.

Pass@k estimator follows Chen et al., Evaluating Large Language Models
Trained on Code (2021). Majority ties are abstentions in this demo.
"""
from collections import Counter
from math import comb


def pass_at_k(n, correct, k):
    if any(type(v) is not int for v in (n, correct, k)):
        raise ValueError("Counts must be integers")
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1 - comb(n - correct, k) / comb(n, k)


def majority_answer(answers):
    if not answers:
        raise ValueError("Need at least one answer")
    counts = Counter(answers).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


if __name__ == "__main__":
    answers, expected = [41, 41, 41, 42, 42], 42
    correct = sum(a == expected for a in answers)
    print("Synthetic answers =", answers, "expected =", expected)
    for k in (1, 2, 5):
        print(f"pass@{k} estimate = {pass_at_k(len(answers), correct, k):.4f}")
    print("consensus@5 answer =", majority_answer(answers))
    print("consensus correct =", majority_answer(answers) == expected)
