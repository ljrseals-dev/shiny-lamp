"""Lazy Cartesian enumeration with independent, resumable sharding.

The engine never holds the full combination list in memory. It streams.

Every combination has a global index in ``range(count(space))``. ``combo_at``
decodes an index straight into a combination (mixed radix, last axis varies
fastest, i.e. the same order as ``itertools.product``), so a shard jumps
directly to its own indices instead of re-walking the whole product.

Hard limit: MAX_COMBINATIONS = 10**9.
"""
from __future__ import annotations

from typing import Any, Iterator

MAX_COMBINATIONS = 10**9


def count(space: dict[str, list]) -> int:
    """Return |A1| * |A2| * ... * |An| without enumerating.

    Cheap even for 10^12 theoretical combinations. Returns 0 if any
    axis is empty.
    """
    total = 1
    for values in space.values():
        if not values:
            return 0
        total *= len(values)
        if total > MAX_COMBINATIONS:
            # Intentional: return the over-cap value (not the exact product) as
            # soon as we know it is over the cap, so callers can format the
            # "exceeds hard limit" error without us multiplying forever.
            return total
    return total


def combo_at(space: dict[str, list], index: int) -> dict[str, Any]:
    """Decode global ``index`` into its combination (mixed-radix, O(#axes)).

    Raises IndexError if ``index`` is outside ``range(count(space))``.
    """
    if index < 0:
        raise IndexError(f"index {index} is negative")
    axes = list(space.keys())
    combo: dict[str, Any] = {}
    for axis in reversed(axes):
        values = space[axis]
        if not values:
            raise IndexError(f"axis '{axis}' is empty; space has no combinations")
        index, pos = divmod(index, len(values))
        combo[axis] = values[pos]
    if index:  # digits left over -> index >= count(space)
        raise IndexError("index is outside the search space")
    return {a: combo[a] for a in axes}


def cartesian(space: dict[str, list]) -> Iterator[dict[str, Any]]:
    """Lazily yield every combination as a dict. Never materialises the set.

    Decodes via ``combo_at`` so there is exactly one decoding path.
    """
    for k in range(count(space)):
        yield combo_at(space, k)


def shard(space: dict[str, list], n_shards: int) -> list[Iterator[dict[str, Any]]]:
    """Split the space by modular global index; each shard does only its own work.

    Shard ``i`` yields the combinations whose global index ``k`` satisfies
    ``k % n_shards == i`` and decodes each one directly with ``combo_at``. It
    does not scan the other shards' combinations, so total work across all
    shards is ``count(space)``, not ``n_shards * count(space)``. Shards are
    independent: a node can restart its shard without coordinating with others.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    total = count(space)

    def _shard_iter(idx: int) -> Iterator[dict[str, Any]]:
        for k in range(idx, total, n_shards):
            yield combo_at(space, k)

    return [_shard_iter(i) for i in range(n_shards)]


def estimate_shard_sizes(space: dict[str, list], n_shards: int) -> list[int]:
    """Return the exact number of combinations each shard will process."""
    total = count(space)
    base = total // n_shards
    remainder = total % n_shards
    return [base + (1 if i < remainder else 0) for i in range(n_shards)]
