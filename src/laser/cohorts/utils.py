"""Utility helpers for the laser.cohorts package."""

from collections.abc import Iterable
from typing import Type

import numpy as np

PropertyType = tuple[str, int, Type[int] | Type[float] | type[np.generic], int | float]

# ---------------------------------------------------------------------------
# Helper: static routing
# ---------------------------------------------------------------------------


def static_routing(routing_2d: np.ndarray, nticks: int) -> np.ndarray:
    """Return a time-invariant 3-D routing view via ``np.broadcast_to``.

    Creates a read-only ``(nticks, nnodes, nnodes)`` view of a 2-D routing
    matrix without allocating a copy.  Pass the result directly to
    ``Migration(..., routing=static_routing(r2d, nticks))``.

    Args:
        routing_2d (np.ndarray): Shape ``(nnodes, nnodes)`` routing matrix
            where ``routing_2d[i, j]`` is the unnormalised weight from node i
            to node j.
        nticks (int): Number of simulation ticks.

    Returns:
        np.ndarray: Read-only shape ``(nticks, nnodes, nnodes)`` broadcast
            view.  No data is copied; memory usage is O(nnodes²).

    Example:
        >>> r2d = np.array([[0, 1], [1, 0]], dtype=np.float64)
        >>> r3d = static_routing(r2d, nticks=365)
        >>> r3d.shape
        (365, 2, 2)
        >>> r3d.base is r2d
        False
        >>> import numpy as np; np.shares_memory(r3d, r2d)
        True
    """
    n = routing_2d.shape[0]
    return np.broadcast_to(routing_2d[None, :, :], (nticks, n, n))


def get_node_mask(
    model,
    nodes: int | np.integer | Iterable[int | np.integer],
) -> int | slice | np.ndarray:
    """Convert a node selector into a numpy-indexable value.

    Returns one of three forms, chosen for performance:

    - ``int`` — when ``nodes`` is a single integer, the index of that node
      (use to drop the node axis when indexing).
    - ``slice(None)`` — when ``nodes`` selects every node in the scenario.
    - boolean ``np.ndarray`` of length ``nnodes`` — otherwise.

    Args:
        model: The parent Model instance (used only to read ``len(scenario)``).
        nodes: A single node id, or any iterable of node ids.

    Returns:
        int | slice | np.ndarray: A value usable as a numpy index along
            the node axis.

    Raises:
        ValueError: If any node id is outside ``[0, nnodes)``.
    """
    nnodes = len(model.scenario)

    if isinstance(nodes, (int, np.integer)) and not isinstance(nodes, bool):
        idx = int(nodes)
        if not 0 <= idx < nnodes:
            raise ValueError(f"node id {idx} out of range [0, {nnodes})")
        return idx

    # Materialise once: avoids the generator double-iteration trap below.
    node_arr = np.fromiter(nodes, dtype=np.int32)
    if node_arr.size and (node_arr.min() < 0 or node_arr.max() >= nnodes):
        raise ValueError(f"node ids {node_arr[(node_arr < 0) | (node_arr >= nnodes)].tolist()} out of range [0, {nnodes})")

    # Fast-path for "all nodes": every id in [0, nnodes) is present (duplicates OK).
    if node_arr.size >= nnodes and np.unique(node_arr).size == nnodes:
        return slice(None)

    mask = np.zeros(nnodes, dtype=bool)
    mask[node_arr] = True

    return mask
