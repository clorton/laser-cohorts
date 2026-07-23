"""Routine immunization component for cohort-based simulation.

Provides ``RoutineImmunization``, a model component that periodically moves a
fraction of susceptibles into a dedicated ``V`` (vaccinated) compartment to
represent ongoing baseline vaccination programmes (e.g. infant immunization).

Unlike the campaign-driven ``Vaccination`` *intervention* (which fires on
specific scheduled ticks under a ``Campaign``), ``RoutineImmunization`` is a
plain component on ``model.components`` and fires on a fixed period throughout
the simulation.

The mathematics follow the user's specification directly. Given an annual
coverage rate ``r`` and an eligible fraction of susceptibles ``ef``, the daily
fraction of susceptibles vaccinated is ``r * ef / 365``.  When the component
fires every ``period`` ticks (instead of every tick), each firing must vaccinate
the cohort accumulated over the preceding ``period`` days, so the per-firing
fraction is ``period * r * ef / 365``.  The number of new vaccinations per node
on each firing is drawn from a Poisson distribution with mean equal to that
fraction times the current susceptible count, then capped at the number of
susceptibles actually available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from laser.cohorts.utils import PropertyType

if TYPE_CHECKING:
    from laser.cohorts.model import Model

logger = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.0


class RoutineImmunization:
    """Routine immunization (RI) — periodic baseline vaccination of susceptibles.

    On each firing tick, draws ``poisson(mean)`` new vaccinations per node,
    where ``mean = (period * coverage * eligible_fraction / 365) * S`` and
    ``S`` is the current per-node susceptible count.  The draw is capped at
    ``S`` (you can never vaccinate more individuals than exist), then those
    individuals are moved from ``S`` to ``V`` and the per-node total is
    accumulated on the ``ri_vaccinated`` node property.

    The component fires on every tick ``t`` for which ``t % period == 0``,
    starting at tick 0.  With ``period=1`` (the default) it fires every tick;
    with ``period=30`` it fires every 30 days; etc.

    Args:
        model (Model): The parent model instance.
        coverage (float): Annual coverage rate ``r`` in ``[0, 1]`` — the
            fraction of the eligible susceptible cohort that the programme
            aims to vaccinate per year.
        eligible_fraction (float): Fraction of susceptibles eligible for
            vaccination ``ef`` in ``[0, 1]``.  A common use is to scale by the
            age-eligible share of the population (e.g. ``ef = 1/70`` for an
            infant-only RI in a population with ~70-year life expectancy).
            Defaults to ``1.0`` — every susceptible is eligible.
        period (int): Tick period between firings.  Must be a positive integer.
            Defaults to ``1`` (fire every tick).

    Raises:
        ValueError: If ``coverage`` is outside ``[0, 1]``.
        ValueError: If ``eligible_fraction`` is outside ``[0, 1]``.
        ValueError: If ``period`` is not a positive integer.

    Example:
        >>> # Infant RI at 80% annual coverage, applied monthly
        >>> ri = RoutineImmunization(
        ...     model,
        ...     coverage=0.8,
        ...     eligible_fraction=1.0 / 70,   # ~infant share of all S
        ...     period=30,
        ... )
        >>> model.components = [
        ...     Susceptible(model),
        ...     Infectious(model, r_recovery=r_recoveries),
        ...     Recovered(model),
        ...     Transmission(model, beta=betas),
        ...     ri,
        ... ]
    """

    def __init__(
        self,
        model: Model,
        coverage: float,
        eligible_fraction: float = 1.0,
        period: int = 1,
    ) -> None:
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"RoutineImmunization: coverage must be in [0, 1], got {coverage}")
        if not 0.0 <= eligible_fraction <= 1.0:
            raise ValueError(f"RoutineImmunization: eligible_fraction must be in [0, 1], got {eligible_fraction}")
        if not isinstance(period, int) or isinstance(period, bool) or period < 1:
            raise ValueError(f"RoutineImmunization: period must be a positive integer, got {period!r}")

        self.model = model
        self.coverage = float(coverage)
        self.eligible_fraction = float(eligible_fraction)
        self.period = period

        # Pre-compute the per-firing fraction so step() is one multiply + Poisson.
        self._fraction_per_firing = self.period * self.coverage * self.eligible_fraction / _DAYS_PER_YEAR
        logger.info(
            "RoutineImmunization: coverage=%.4f eligible_fraction=%.4f period=%d → fraction_per_firing=%.6f",
            self.coverage,
            self.eligible_fraction,
            self.period,
            self._fraction_per_firing,
        )

        return

    def setup(self) -> None:
        """No-op setup hook — V is allocated automatically from `states`."""
        return

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        return

    def step(self, tick: int) -> None:
        """Apply one round of routine immunization on firing ticks.

        Skips the tick unless ``tick % period == 0``.  On firing ticks,
        Poisson-draws ``mean = fraction_per_firing * S`` new vaccinations per
        node, caps each per-node draw at the current susceptible count, moves
        the drawn individuals from ``S`` to ``V`` at ``tick + 1``, and
        accumulates the per-node count in ``nodes.ri_vaccinated[tick]``.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        if tick % self.period != 0:
            return

        S = self.model.states.S[tick + 1]  # (nnodes,) view
        mean = self._fraction_per_firing * S.astype(np.float64)
        draws = np.random.poisson(mean).astype(np.int32)
        # Cap at available susceptibles — the Poisson can over-shoot for
        # large fractions, and we can't vaccinate more individuals than exist.
        draws = np.minimum(draws, S.astype(np.int32))

        self.model.states.S[tick + 1] -= draws
        self.model.states.V[tick + 1] += draws
        self.model.nodes.ri_vaccinated[tick] += draws

        return

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        return

    @property
    def properties(self) -> list[PropertyType]:
        """Return the node properties required by this component.

        Returns:
            list[PropertyType]: ``[("ri_vaccinated", nticks, np.int32, 0)]``
        """
        return [("ri_vaccinated", int(self.model.params.nticks), np.int32, 0)]  # type: ignore

    @property
    def states(self) -> list[str]:
        """Return the compartment states declared by this component.

        Returns:
            list[str]: ``["V"]`` — the routine-immunization vaccinated state.
        """
        return ["V"]
