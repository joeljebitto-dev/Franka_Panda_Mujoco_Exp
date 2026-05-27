from __future__ import annotations

from collections.abc import Sequence


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def clamp_vector(values: Sequence[float], limits: Sequence[tuple[float, float]]) -> list[float]:
    if len(values) != len(limits):
        raise ValueError(f"Expected {len(limits)} values, got {len(values)}")
    return [clamp(float(value), low, high) for value, (low, high) in zip(values, limits)]
