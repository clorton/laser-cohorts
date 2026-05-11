"""Vital dynamics components for cohort-based simulation.

Provides NonDiseaseMortality, a component that applies background (non-disease)
mortality to one or more compartment states each simulation tick.
"""

from collections.abc import Iterable

import numpy as np
from laser.generic.utils import ValuesMap

from laser.cohorts.model import Model
from laser.cohorts.utils import PropertyType


class NonDiseaseMortality:
    """Background (non-disease) mortality component.

    Applies a binomial mortality draw each tick to the specified compartment
    states, removing individuals and recording the deaths in
    ``nodes.non_disease_mortality``.

    By default acts on every state in ``model.states``.  Passing ``states``
    restricts mortality to only the named compartments, creating an implicit
    mask over the state axis.

    Example:
        >>> ndm = NonDiseaseMortality(model, mu=1/365/70)  # ~70-year life expectancy
        >>> ndm_s_only = NonDiseaseMortality(model, mu=1/365/70, states=["S"])
    """

    def __init__(
        self,
        model: Model,
        mu: int | float | ValuesMap | np.ndarray,
        states: Iterable[str] | None = None,
    ) -> None:
        """Initialize the NonDiseaseMortality component.

        Args:
            model (Model): The parent model instance.
            mu (int | float | ValuesMap | np.ndarray): Per-tick, per-node crude
                mortality rate.  A scalar is broadcast to all ticks and nodes via
                ``ValuesMap.from_scalar``; a ``ValuesMap`` or 2-D array of shape
                ``(nticks, nnodes)`` is used directly.
            states (Iterable[str] | None): Names of compartment states to apply
                mortality to.  Accepts any iterable (list, tuple, set, generator,
                etc.).  ``None`` applies mortality to every state in
                ``model.states``.
        """
        self.model = model
        if np.isscalar(mu):
            self.mu = ValuesMap.from_scalar(mu, model.params.nticks, len(model.scenario))
        else:
            self.mu = mu
        self._requested_states = set(states) if states is not None else None
        self._state_views: list[np.ndarray] = []

    def setup(self) -> None:
        """Cache plain-ndarray views for the target compartment states.

        Resolves which states to apply mortality to — all states when
        ``states`` was ``None``, or the requested subset — then caches a view
        for each so that ``step`` can modify them in place without repeated
        attribute lookups.
        """
        all_names = self.model.states.state_names or ()
        if self._requested_states is None:
            active = list(all_names)
        else:
            active = [n for n in all_names if n in self._requested_states]
        self._state_views = [getattr(self.model.states, name) for name in active]

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Apply binomial mortality draws to all target states.

        For each target compartment, converts the mortality rate to a per-tick
        survival probability via ``-expm1(-mu)``, draws deaths from a binomial
        distribution, subtracts them from the compartment at ``tick+1``, and
        accumulates them in ``nodes.non_disease_mortality``.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        probability = -np.expm1(-self.mu[tick])
        for state_view in self._state_views:
            mortality = np.random.binomial(state_view[tick + 1], probability).astype(np.int32)
            self.model.nodes.non_disease_mortality[tick] += mortality
            state_view[tick + 1] -= mortality

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return the node properties required by this component.

        Returns:
            list[PropertyType]: ``[("non_disease_mortality", nticks, np.int32, 0)]``
        """
        return [("non_disease_mortality", self.model.params.nticks, np.int32, 0)]

    @property
    def states(self) -> list[str]:
        """Return the compartment states created by this component.

        Returns:
            list[str]: Empty list; this component acts on existing states but
                does not create any new ones.
        """
        return []
