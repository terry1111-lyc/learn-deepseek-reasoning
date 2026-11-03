"""Scalar toy experts: routing selection and mixture weights are distinct.

Not V3 inference: omits expert groups, neural layers, distributed operations,
route scaling and learned bias updates. Scores and experts are invented.
"""
import math


def route(scores, biases, k):
    if not scores or len(scores) != len(biases) or not 1 <= k <= len(scores):
        raise ValueError("Invalid scores, biases or k")
    if not all(math.isfinite(v) for v in [*scores, *biases]):
        raise ValueError("Values must be finite")
    if any(s < 0 for s in scores):
        raise ValueError("Scores must be nonnegative")
    # Bias affects selection; mixing weights use ORIGINAL scores.
    selected = sorted(range(len(scores)),
                      key=lambda i: (-(scores[i] + biases[i]), i))[:k]
    total = sum(scores[i] for i in selected)
    if total == 0:
        raise ValueError("Selected scores must have positive total")
    return selected, [scores[i] / total for i in selected]


def toy_moe(x, scores, biases, k):
    if len(scores) != 4:
        raise ValueError("This demo has exactly four toy experts")
    selected, weights = route(scores, biases, k)
    experts = (lambda t: t, lambda t: 2*t, lambda t: -t, lambda t: t+1)
    shared = x * 0.1
    output = shared + sum(w * experts[i](x) for i, w in zip(selected, weights))
    return selected, weights, output


if __name__ == "__main__":
    scores = [0.6, 0.3, 0.2, 0.1]
    for biases in ([0, 0, 0, 0], [0, 0, 0.5, 0]):
        selected, weights, output = toy_moe(2, scores, biases, 2)
        print("biases =", biases, "selected =", selected,
              "weights =", [round(w, 4) for w in weights],
              "output =", round(output, 4))
