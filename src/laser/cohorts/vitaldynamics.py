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
        >>> ndm = NonDiseaseMortality(model, r_mortality=1/365/70)  # ~70-year life expectancy
        >>> ndm_s_only = NonDiseaseMortality(model, r_mortality=1/365/70, states=["S"])
    """

    def __init__(
        self,
        model: Model,
        r_mortality: int | float | ValuesMap | np.ndarray,
        states: Iterable[str] | None = None,
    ) -> None:
        """Initialize the NonDiseaseMortality component.

        Args:
            model (Model): The parent model instance.
            r_mortality (int | float | ValuesMap | np.ndarray): Per-tick, per-node
                crude mortality rate.  A scalar is broadcast to all ticks and nodes
                via ``ValuesMap.from_scalar``; a ``ValuesMap`` or 2-D array of shape
                ``(nticks, nnodes)`` is used directly.
            states (Iterable[str] | None): Names of compartment states to apply
                mortality to.  Accepts any iterable (list, tuple, set, generator,
                etc.).  ``None`` applies mortality to every state in
                ``model.states``.
        """
        self.model = model
        if np.isscalar(r_mortality):
            self.r_mortality = ValuesMap.from_scalar(r_mortality, model.params.nticks, len(model.scenario))  # type: ignore
        else:
            self.r_mortality = r_mortality
        self._requested_states = set(states) if states is not None else None
        self._state_mask: np.ndarray | slice | None = None

        return

    def setup(self) -> None:
        """Build a boolean state mask for the target compartment states.

        Resolves which states to apply mortality to — all states when
        ``states`` was ``None``, or the requested subset — then builds a
        boolean mask over the state axis using ``get_state_index`` so that
        ``step`` can select all target states in one vectorised operation.
        """
        # None means all and all means all
        if (self._requested_states is None) or (set(self._requested_states) == set(self.model.states.state_names)):  # type: ignore
            mask = slice(None)  # equivalent to `:`
        else:
            mask = self.model.states.get_state_mask(list(self._requested_states))
        self._state_mask = mask

        return

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Apply binomial mortality draws to all target states in one operation.

        Selects all target compartments at ``tick+1`` via the boolean state
        mask, draws deaths from a binomial distribution for all of them at
        once, subtracts the deaths back via boolean-index assignment, and
        accumulates per-node totals in ``nodes.non_disease_mortality``.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        probability = -np.expm1(-self.r_mortality[tick])  # type: ignore
        states_at_tick = self.model.states[tick + 1]  # view: (nstates, nnodes)
        active = states_at_tick[self._state_mask]  # copy: (n_active, nnodes)
        mortality = np.random.binomial(active, probability).astype(np.int32)
        # states_at_tick reduces dimensionality by 1
        axis = self.model.states.state_axis - 1
        self.model.nodes.non_disease_mortality[tick] += mortality.sum(axis=axis)
        states_at_tick[self._state_mask] -= mortality

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
        return [("non_disease_mortality", self.model.params.nticks, np.int32, 0)]  # type: ignore

    @property
    def states(self) -> list[str]:
        """Return the compartment states created by this component.

        Returns:
            list[str]: Empty list; this component acts on existing states but
                does not create any new ones.
        """
        return []


class ConstantPopBirths:
    """Constant-population birth component.

    Reads the per-node death count recorded by ``NonDiseaseMortality`` at each
    tick and adds the same number of individuals back into the S compartment,
    keeping the total population constant across ticks.

    Must be ordered *after* ``NonDiseaseMortality`` in the component list so
    that ``non_disease_mortality`` is populated before births are applied.

    Example:
        >>> ndm = NonDiseaseMortality(model, r_mortality=1/365/70)
        >>> cpb = ConstantPopBirths(model)
        >>> model.components = [Susceptible(model), ndm, cpb]
    """

    def __init__(self, model: Model) -> None:
        """Initialize the ConstantPopBirths component.

        Args:
            model (Model): The parent model instance.
        """
        self.model = model

    def setup(self) -> None:
        """No-op setup hook."""
        pass

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Add births equal to deaths recorded this tick into the S compartment.

        Reads ``nodes.non_disease_mortality[tick]`` — the deaths accumulated by
        ``NonDiseaseMortality`` during the current tick — and adds that count
        to ``states.S[tick+1]``, replacing every death with one new susceptible.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        births = self.model.nodes.non_disease_mortality[tick]
        self.model.states.S[tick + 1] += births

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
        return [("non_disease_mortality", self.model.params.nticks, np.int32, 0)]  # type: ignore

    @property
    def states(self) -> list[str]:
        """Return the compartment states used by this component.

        Returns:
            list[str]: ``["S"]``
        """
        return ["S"]
