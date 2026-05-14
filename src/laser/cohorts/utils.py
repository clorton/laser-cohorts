"""Utility helpers for the laser.cohorts package."""

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
