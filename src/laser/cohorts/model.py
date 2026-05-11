"""Core Model class for cohort-based compartmental disease simulation."""

import numpy as np

from geopandas.geodataframe import GeoDataFrame
from typing import Optional
from laser.core import PropertySet
from laser.core import LaserFrame

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
    """

    def __init__(self, scenario: GeoDataFrame, params: Optional[PropertySet] = None) -> None:
        """Initialize the Model.

        Args:
            scenario (GeoDataFrame): Geographic/demographic scenario; each row is a node.
            params (PropertySet | None): Named model parameters. Must contain `nticks`
                before `components` is assigned.
        """
        self.scenario = scenario
        self.params = params
        self.nodes = LaserFrame(len(scenario))
        self._components = []
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

        for component in self.components:
            component.setup()

        return

    def run(self) -> None:
        """Execute the simulation for `params.nticks` time steps.

        For each tick, calls `start_step`, `step`, and `end_step` on every
        registered component in order.
        """
        for tick in range(self.params.nticks):
            for component in self.components:
                component.start_step(tick)
            for component in self.components:
                component.step(tick)
            for component in self.components:
                component.end_step(tick)
        return
