"""Core Model class for cohort-based compartmental disease simulation."""

import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from laser.core import LaserFrame
from laser.core import PropertySet

from laser.cohorts.statearray import StateArray


class Model:
    """Compartmental disease model that orchestrates cohort-based simulation components.

    Manages a population scenario, a set of epidemiological components, and the
    discrete-time simulation loop.

    Attributes:
        scenario (GeoDataFrame): Geographic/demographic scenario defining nodes.
        params (PropertySet | None): Model parameters, including `nticks`.
        nodes (LaserFrame): Per-node property storage.
        states (StateArray): Compartment state array of shape
            (nticks+1, n_states, n_nodes).
        network (np.ndarray): 2-D inter-node mixing matrix of shape
            (nnodes, nnodes).  Defaults to a uniform-zero scalar set at
            initialisation.
    """

    def __init__(
        self,
        scenario: GeoDataFrame,
        params: Optional[PropertySet] = None,
        carry_forward_states: Iterable[str] | None = None,
    ) -> None:
        """Initialize the Model.

        Args:
            scenario (GeoDataFrame): Geographic/demographic scenario; each row is a node.
            params (PropertySet | None): Named model parameters. Must contain `nticks`
                before `components` is assigned.
            carry_forward_states (Iterable[str] | None): Names of compartment states to
                carry forward at the start of each tick.  ``None`` (default) carries
                forward every state.  Pass an explicit iterable to restrict carry-forward
                to a subset.
        """
        self.scenario = scenario
        self.params = params
        self.nodes = LaserFrame(len(scenario))
        self._components = []
        self._carry_forward_states = set(carry_forward_states) if carry_forward_states is not None else None
        self._carry_mask: np.ndarray | slice = slice(None)
        self.network = np.zeros((len(scenario), len(scenario)), dtype=np.float32)

        return

    @property
    def components(self) -> list:
        """Return the list of registered simulation components.

        Returns:
            list: Ordered list of component instances.
        """
        return list(self._components)

    @components.setter
    def components(self, proposal: list) -> None:
        """Set model components and initialize states and node properties.

        Collects unique state names and node properties from all components,
        allocates the state array and node property arrays, then calls `setup()`
        on each component.

        Args:
            proposal (list): Ordered list of component instances to register.
        """
        self._components = proposal

        states = []
        properties = []
        for component in self._components:
            for state in component.states:
                if state not in states:
                    states += [state]
            for property in component.properties:
                if property not in properties:
                    properties += [property]

        self.states = StateArray(
            states, 1, shape=(self.params.nticks + 1, len(states), len(self.scenario)), dtype=np.int32, default_value=0
        )
        for name, count, dtype, default in properties:
            self.nodes.add_array_property(name, shape=(self.params.nticks, len(self.scenario)), dtype=dtype, default=default)

        if self._carry_forward_states is not None:
            all_names = self.states.state_names or ()
            mask = np.zeros(len(all_names), dtype=bool)
            for name in self._carry_forward_states:
                idx = self.states.get_state_index(name)
                if idx is not None:
                    mask[idx] = True
                else:
                    warnings.warn(
                        f"carry_forward_states: '{name}' is not a registered state and will be ignored.",
                        UserWarning,
                        stacklevel=3,
                    )
            self._carry_mask = mask

        for component in self.components:
            component.setup()

        return

    @property
    def network(self) -> np.ndarray:
        """Return the inter-node mixing matrix.

        Returns:
            np.ndarray: 2-D array of shape ``(nnodes, nnodes)``
                whose entry ``[i, j]`` gives the connectivity weight from node
                ``i`` to node ``j``.
        """
        return self._network

    @network.setter
    def network(self, value) -> None:
        """Set the inter-node mixing matrix.

        Must already be 2-D and exactly ``(nnodes, nnodes)`` in shape.

        Args:
            value (np.ndarray): Connectivity weights between nodes..

        Raises:
            ValueError: If ``value`` is not 2-D.
            ValueError: If ``value.shape`` is not ``(nnodes, nnodes)``.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"network must be a NumPy array, got {type(value)}")
        if value.shape != (len(self.scenario), len(self.scenario)):
            raise ValueError(f"network must be shape {(len(self.scenario), len(self.scenario))}, got {value.shape}")
        self._network = value

        return

    def run(self) -> None:
        """Execute the simulation for `params.nticks` time steps.

        At the start of each tick, carries forward the selected compartment
        states from tick to tick+1, then calls `start_step`, `step`, and
        `end_step` on every registered component in order.
        """
        for tick in range(self.params.nticks):
            cur = self.states[tick]
            nxt = self.states[tick + 1]
            nxt[self._carry_mask] = cur[self._carry_mask]

            # for component in self.components:
            #     component.start_step(tick)
            #     component.step(tick)
            #     component.end_step(tick)

            for component in self.components:
                component.start_step(tick)
            for component in self.components:
                component.step(tick)
            for component in self.components:
                component.end_step(tick)
        return
