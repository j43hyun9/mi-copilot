"""Staircase grokking detection (Charton 2024)."""
from __future__ import annotations


def detect_staircase(entries: list[dict], threshold: float = 50.0,
                     gcd_max: int = 30) -> list[tuple[int, set[int]]]:
    """Detect epochs where new GCD values were learned (acc crossed threshold).

    A "staircase" grokking pattern: model learns to predict GCD k for new
    values of k at discrete epochs, one (or a few) at a time.

    Args:
        entries: list of dicts, each with "epoch" + "test_arithmetic_acc_{k}" keys
                 (parsed from Charton's train.log __log__ entries)
        threshold: accuracy % at which a GCD is considered "learned"
        gcd_max: max GCD value to track

    Returns:
        list of (epoch, set_of_newly_learned_gcds) tuples, in chronological order.
        Only epochs where new GCDs crossed threshold are returned.

    Example:
        >>> # parse log into entries first, then:
        >>> staircase = detect_staircase(entries, threshold=50.0)
        >>> for ep, new in staircase:
        ...     print(f"epoch {ep}: learned GCD {sorted(new)}")
    """
    transitions: list[tuple[int, set[int]]] = []
    prev_learned: set[int] = set()
    for e in entries:
        now_learned: set[int] = set()
        for k in range(gcd_max + 1):
            v = e.get(f"test_arithmetic_acc_{k}")
            if v is not None and v >= threshold:
                now_learned.add(k)
        new = now_learned - prev_learned
        if new:
            transitions.append((e["epoch"], new))
        prev_learned = now_learned
    return transitions


def parse_charton_log(log_path: str) -> list[dict]:
    """Parse Charton's train.log file, extracting __log__ JSON entries.

    Each line containing "__log__:{...}" is parsed as JSON.

    Args:
        log_path: filesystem path to train.log

    Returns:
        list of dicts (one per epoch)
    """
    import json
    import re

    entries: list[dict] = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"__log__:({.+})", line)
            if m:
                entries.append(json.loads(m.group(1)))
    return entries
