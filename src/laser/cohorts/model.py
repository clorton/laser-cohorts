import numpy as np

from geopandas.geodataframe import GeoDataFrame
from typing import Optional
from laser.core import PropertySet
from laser.core import LaserFrame

from laser.cohorts.statearray import StateArray


class Model:
    def __init__(self, scenario: GeoDataFrame, params: Optional[PropertySet] = None) -> None:
        self.scenario = scenario
        self.params = params
        self.nodes = LaserFrame(len(scenario))
        self._components = []
        return

    @property
    def components(self) -> list:
        return self._components

    @components.setter
    def components(self, proposal) -> None:
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
        for tick in range(self.params.nticks):
            for component in self.components:
                component.start_step(tick)
            for component in self.components:
                component.step(tick)
            for component in self.components:
                component.end_step(tick)
        return
