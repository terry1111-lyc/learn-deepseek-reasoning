"""Group-relative advantages and clipping, not a GRPO trainer.

Original educational implementation. No model, gradients, or KL computation.
"""
import math


def advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Need at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std == 0:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + 1e-8) for r in rewards]


def clipped_surrogate(ratio, advantage, epsilon=0.2):
    if not all(math.isfinite(x) for x in (ratio, advantage, epsilon)):
        raise ValueError("Inputs must be finite")
    if ratio < 0 or not 0 <= epsilon < 1:
        raise ValueError("Invalid probability ratio or clipping epsilon")
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


if __name__ == "__main__":
    for rewards in ([1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]):
        print("rewards =", rewards, "advantages =",
              [round(a, 4) for a in advantages(rewards)])
    for advantage in (1.0, -1.0):
        print("ratio=1.5, advantage=", advantage,
              "surrogate=", clipped_surrogate(1.5, advantage))
