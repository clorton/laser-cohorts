"""Stochastic migration component for cohort-based simulation.

Moves individuals between nodes each tick by drawing emigrants from each
compartment via a binomial draw and then routing them to destination nodes
via a sequential-binomial decomposition of the multinomial.

The routing matrix is 3-D with shape ``(nticks, nnodes, nnodes)`` so that
connectivity can vary over time.  For static connectivity, use
``np.broadcast_to(routing_2d[None], (nticks, nnodes, nnodes))`` to obtain
a read-only 3-D view without copying data.
"""

from typing import TYPE_CHECKING

import numpy as np
from laser.generic.utils import ValuesMap

if TYPE_CHECKING:
    from laser.cohorts.model import Model

from laser.cohorts.utils import PropertyType


class Migration:
    """Stochastic inter-node migration component with time-varying routing.

    Each tick, applies a per-node emigration rate (converted to probability)
    uniformly across every compartment state.  For each source node the total
    number of emigrants per compartment is drawn from a binomial distribution,
    then those emigrants are distributed to destination nodes via a sequential-
    binomial decomposition of the multinomial.

    The routing tensor is 3-D so connectivity can vary tick-by-tick.  Pass a
    ``np.broadcast_to`` view to represent static routing without copying:

        routing_3d = np.broadcast_to(routing_2d[None], (nticks, n, n))

    Population is conserved: every emigrant from node i is assigned to exactly
    one destination node j (including possibly i itself if the routing diagonal
    is non-zero, which has no net effect).

    Attributes:
        model (Model): The parent model instance.
        r_migration (ValuesMap): Per-tick, per-node emigration rate.
        routing (np.ndarray): Row-normalised routing tensor of shape
            ``(nticks, nnodes, nnodes)``; entry ``[t, i, j]`` is the fraction
            of emigrants from node i going to node j on tick t.

    Raises:
        ValueError: If ``routing.ndim != 3`` or
            ``routing.shape != (nticks, nnodes, nnodes)``.
    """

    def __init__(
        self,
        model: "Model",
        r_migration: int | float | ValuesMap | np.ndarray,
        routing: np.ndarray,
    ) -> None:
        """Initialize the Migration component.

        Args:
            model (Model): The parent model instance.
            r_migration (int | float | ValuesMap | np.ndarray): Per-tick,
                per-node emigration rate.  Scalars are broadcast to all ticks
                and nodes via ``ValuesMap.from_scalar``.
            routing (np.ndarray): Shape ``(nticks, nnodes, nnodes)`` tensor
                where ``routing[t, i, j]`` is the unnormalised weight of
                migration from node i to node j on tick t.  Each row is
                normalised internally.  Rows summing to zero do not emigrate
                on that tick.  For static routing pass a ``np.broadcast_to``
                view:

                    np.broadcast_to(routing_2d[None], (nticks, n, n))

        Raises:
            ValueError: If ``routing.shape`` is not ``(nticks, nnodes, nnodes)``.
        """
        n = len(model.scenario)
        nticks = model.params.nticks
        expected = (nticks, n, n)
        if routing.shape != expected:
            raise ValueError(
                f"routing must be shape {expected}, got {routing.shape}. "
                f"For static 2-D routing use: "
                f"np.broadcast_to(routing_2d[None], {expected})"
            )

        self.model = model
        if np.isscalar(r_migration):
            self.r_migration = ValuesMap.from_scalar(r_migration, nticks, n)
        else:
            self.r_migration = r_migration

        # Normalise rows: routing[t, i, :] sums to 1 for non-zero rows.
        # np.broadcast_to produces a read-only array, so we must create a new
        # writable array here rather than modifying in-place.
        row_sums = routing.sum(axis=2, keepdims=True)  # (nticks, nnodes, 1)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        self._routing = np.where(row_sums > 0, routing / safe_sums, 0.0)
        self._emigrates = row_sums.squeeze(axis=2) > 0  # (nticks, nnodes) bool

    def setup(self) -> None:
        """No-op; migration requires no additional initialisation."""
        pass

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Move individuals between nodes according to r_migration and routing.

        Steps:

        1. Convert emigration rate to probability via ``-expm1(-r)``.
        2. Zero out probability for nodes with no routing row this tick.
        3. Draw total emigrants per compartment via a single vectorised
           binomial over ``(nstates, nnodes)``.
        4. Distribute emigrants to destinations using a sequential-binomial
           decomposition of the multinomial: for each destination j, draw
           ``binomial(remaining, conditional_fraction)`` over all sources
           simultaneously, accumulate inflow, and reduce ``remaining``.
           The last destination receives whatever is left over, preserving
           the exact population count.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        emigrates_tick = self._emigrates[tick]  # (nnodes,)
        routing_tick = self._routing[tick]  # (nnodes, nnodes)

        r = self.r_migration[tick]  # (nnodes,)
        # Zero out probability for non-emigrating nodes to prevent subtracting
        # leavers from nodes whose emigrants are never redistributed.
        p_leave = -np.expm1(-r) * emigrates_tick  # (nnodes,)

        states_now = self.model.states[tick + 1]  # (nstates, nnodes)
        nstates, nnodes = states_now.shape

        leavers = np.random.binomial(states_now, p_leave[None, :]).astype(np.int32)

        # Sequential-binomial decomposition of the multinomial.
        # For each destination j we draw the conditional fraction of the
        # remaining pool, vectorised over all source nodes simultaneously.
        remaining = leavers.copy()  # (nstates, nnodes)
        weight_remaining = routing_tick.sum(axis=1).copy()  # (nnodes,)
        inflow = np.zeros_like(states_now)

        for j in range(nnodes - 1):
            safe_weights = np.maximum(weight_remaining, 1e-12)
            raw_frac = routing_tick[:, j] / safe_weights  # (nnodes,)
            frac = np.where(
                weight_remaining > 0,
                np.clip(raw_frac, 0.0, 1.0),
                0.0,
            )  # (nnodes,)
            drawn = np.random.binomial(remaining, frac[None, :]).astype(np.int32)
            inflow[:, j] += drawn.sum(axis=1)
            remaining -= drawn
            weight_remaining -= routing_tick[:, j]

        # Last destination absorbs the remainder (exact population conservation).
        inflow[:, nnodes - 1] += remaining.sum(axis=1)

        states_now -= leavers
        states_now += inflow

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; migration requires no extra
                node properties.
        """
        return []

    @property
    def states(self) -> list[str]:
        """Return compartment state names required by this component.

        Returns:
            list[str]: Empty list; migration operates on all existing states
                without declaring new ones.
        """
        return []
