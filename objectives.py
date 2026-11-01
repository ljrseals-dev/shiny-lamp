"""Named objective functions. Never accept caller-supplied code.

Objectives are referenced by name from this registry only. This is a
hard safety requirement: the plugin must never eval or exec user strings.
"""
from __future__ import annotations

import math
from typing import Any, Callable

ObjectiveFn = Callable[[dict[str, Any]], float]


def _sphere(combo: dict[str, Any]) -> float:
    """Negative sum of squares. Maximum is at the origin (all zeros)."""
    return -sum(float(v) ** 2 for v in combo.values())


def _rastrigin(combo: dict[str, Any]) -> float:
    """Negative Rastrigin. Global maximum at the origin."""
    vals = [float(v) for v in combo.values()]
    n = len(vals)
    return -(10 * n + sum(v**2 - 10 * math.cos(2 * math.pi * v) for v in vals))


def _rosenbrock(combo: dict[str, Any]) -> float:
    """Negative Rosenbrock (2-D friendly). Higher is better."""
    vals = [float(v) for v in combo.values()]
    if len(vals) < 2:
        return -float(vals[0]) ** 2 if vals else 0.0
    total = 0.0
    for i in range(len(vals) - 1):
        total += 100 * (vals[i + 1] - vals[i] ** 2) ** 2 + (1 - vals[i]) ** 2
    return -total


def _linear_sum(combo: dict[str, Any]) -> float:
    """Simple sum — useful for testing ranking order."""
    return sum(float(v) for v in combo.values())


def _product(combo: dict[str, Any]) -> float:
    """Product of values — useful for testing."""
    prod = 1.0
    for v in combo.values():
        prod *= float(v)
    return prod


REGISTRY: dict[str, ObjectiveFn] = {
    "sphere": _sphere,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "linear_sum": _linear_sum,
    "product": _product,
}


def get(name: str) -> ObjectiveFn:
    """Lookup an objective by name. Raises KeyError with available names."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return REGISTRY[name]


def list_objectives() -> list[str]:
    """Return the names of every registered objective."""
    return sorted(REGISTRY.keys())
