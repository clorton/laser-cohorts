"""Vaccination intervention for the laser.cohorts Campaign component.

Moves a binomial-drawn fraction of each targeted compartment state into a
dedicated ``V`` (vaccinated) state.  Coverage is applied as a direct
probability: ``coverage=0.8`` means each targeted individual has an 80%
chance of being vaccinated on the scheduled tick.
"""

from __future__ import annotations

import logging

import numpy as np

from laser.cohorts.campaign import Intervention
from laser.cohorts.utils import PropertyType

logger = logging.getLogger(__name__)


class Vaccination(Intervention):
    """Move a coverage fraction of targeted individuals into the V compartment.

    Reads ``coverage`` from the ``params`` dict (default 0.0) and applies a
    binomial draw to each targeted state in the targeted nodes.  Results are
    accumulated in the ``newly_vaccinated`` node property.

    Expected ``params`` keys:
        coverage (float): Probability in [0, 1] that any individual in a
            targeted state/node is vaccinated this tick.  Default ``0.0``.

    Raises:
        ValueError: If ``coverage`` is outside [0, 1].

    Example:
        >>> Campaign.register(Vaccination)
        >>> schedule = [
        ...     {"who": ["S"], "what": "Vaccination", "when": 30,
        ...      "where": "*", "parameters": {"coverage": 0.8}, "notes": ""},
        ... ]
    """

    @property
    def states(self) -> list[str]:
        """Return the compartment states declared by this intervention.

        Returns:
            list[str]: ``["V"]``
        """
        return ["V"]

    @property
    def properties(self) -> list[PropertyType]:
        """Return the node properties declared by this intervention.

        Returns:
            list[PropertyType]: ``[("newly_vaccinated", nticks, np.int32, 0)]``
        """
        return [("newly_vaccinated", int(self.model.params.nticks), np.int32, 0)] # type: ignore  # pyright: ignore[reportOptionalMemberAccess]

    def apply(
        self,
        tick: int,
        who: list[str] | None,
        where: list[int] | None,
        params: dict,
        notes: str,
    ) -> None:
        """Vaccinate a fraction of targeted individuals on this tick.

        For each targeted state in each targeted node, draws the number of
        vaccinees from a binomial distribution using ``coverage`` as the
        success probability, removes them from their current state, and adds
        them to ``V[tick + 1]``.

        Args:
            tick (int): Current simulation tick (0-indexed).
            who (list[str] | None): Compartment state names to vaccinate from;
                ``None`` means all states registered in the model.
            where (list[int] | None): Node IDs to vaccinate; ``None`` means
                all nodes.
            params (dict): Must contain ``"coverage"`` (float in [0, 1]).
                Defaults to ``0.0`` if absent.
            notes (str): Free-text annotation; not used by this intervention.

        Raises:
            ValueError: If ``coverage`` is not in the range [0, 1].
        """
        coverage = float(params.get("coverage", 0.0))
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"Vaccination: coverage must be in [0, 1], got {coverage}")

        all_state_names = list(self.model.states.state_names or [])
        target_states = who if who is not None else all_state_names

        nnodes = len(self.model.scenario)
        target_nodes = where if where is not None else list(range(nnodes))

        # Build a per-node probability vector: coverage for targeted nodes, 0 elsewhere.
        node_mask = np.zeros(nnodes, dtype=bool)
        for node in target_nodes:
            node_mask[node] = True
        p = np.where(node_mask, coverage, 0.0)

        states_next = self.model.states[tick + 1]  # (nstates, nnodes)
        V = self.model.states.V
        total_vaccinated = np.zeros(nnodes, dtype=np.int32)

        for state_name in target_states:
            idx = self.model.states.get_state_index(state_name)
            if idx is None:
                logger.warning("Vaccination: state '%s' not found in model; skipping", state_name)
                continue
            state_row = states_next[idx]  # (nnodes,) view
            drawn = np.random.binomial(state_row, p).astype(np.int32)
            states_next[idx] -= drawn
            V[tick + 1] += drawn
            total_vaccinated += drawn

        self.model.nodes.newly_vaccinated[tick] += total_vaccinated
        logger.info(
            "Vaccination tick %d: vaccinated %d total across %d nodes",
            tick,
            int(total_vaccinated.sum()),
            len(target_nodes),
        )
