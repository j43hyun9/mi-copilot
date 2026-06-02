"""Sieve algorithm analysis for GCD models (Charton 2024)."""
from __future__ import annotations


def base_divisors(base: int, max_gcd: int = 1000) -> list[int]:
    """Divisors of base, up to max_gcd. The "sieve-learnable" set.

    Charton shows that a GCD model trained in base B can only learn GCDs that
    are divisors of B (because the digits-pattern signal exists only for those).
    For base 1000 = 2³ × 5³, learnable = {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, ...}.

    Args:
        base: number base used by the model (typically 1000)
        max_gcd: upper bound on GCDs to consider

    Returns:
        sorted list of divisors of `base` that are ≤ max_gcd.
    """
    return sorted(g for g in range(1, max_gcd + 1) if base % g == 0)


def split_learned_vs_unlearnable(per_gcd_acc: dict[int, float], base: int,
                                  threshold: float = 50.0
                                  ) -> tuple[list[int], list[int], list[int]]:
    """Split per-GCD accuracies into three categories.

    Args:
        per_gcd_acc: {gcd_value: accuracy %}
        base: model's number base
        threshold: % considered "learned"

    Returns:
        (learned_divisors, missed_divisors, grokked_primes):
            learned_divisors  — base-divisors that the model learned (expected)
            missed_divisors   — base-divisors model failed on (paper-internal anomaly)
            grokked_primes    — non-base-divisors model still learned (Charton's "prime grokking")
    """
    divs = set(base_divisors(base, max_gcd=max(per_gcd_acc.keys())))
    learned_divs, missed_divs, grokked_primes = [], [], []
    for g, acc in sorted(per_gcd_acc.items()):
        is_div = g in divs
        is_learned = acc >= threshold
        if is_div and is_learned:
            learned_divs.append(g)
        elif is_div and not is_learned:
            missed_divs.append(g)
        elif not is_div and is_learned:
            grokked_primes.append(g)
    return learned_divs, missed_divs, grokked_primes
