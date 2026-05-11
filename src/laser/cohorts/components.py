from typing import Optional, Type

import numpy as np
from laser.generic.utils import ValuesMap

PropertyType = tuple[str, int, Type[int] | Type[float] | np.dtype, int | float]


class Susceptible:
    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        self.validating = validating

        return

    def setup(self) -> None:
        self.model.states.S[0] = self.model.scenario.S
        return

    def start_step(self, tick: int) -> None:
        sus = self.model.states.S
        sus[tick + 1] = sus[tick]
        return

    def step(self, tick: int) -> None:
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        sus = self.model.states.S
        mortality = np.random.binomial(sus[tick + 1], probability).astype(sus.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        sus[tick + 1] -= mortality

        return

    def end_step(self, tick: int) -> None:
        pass

    @property
    def properties(self) -> list[PropertyType]:
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return ["S"]


class Exposed:
    def __init__(self, model, sigma: ValuesMap, mu: Optional[ValuesMap] = None, validating: bool = False):
        self.model = model
        self.sigma = sigma
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        self.validating = validating

        return

    def setup(self) -> None:
        self.model.states.E[0] = self.model.scenario.E
        return

    def start_step(self, tick: int) -> None:
        exp = self.model.states.E
        exp[tick + 1] = exp[tick]
        return

    def step(self, tick: int) -> None:
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        exp = self.model.states.E
        inf = self.model.states.I
        mortality = np.random.binomial(exp[tick + 1], probability).astype(exp.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        exp[tick + 1] -= mortality

        # Disease progression - infectiousness
        probability = -np.expm1(-self.sigma[tick])
        newly_infectious = np.random.binomial(exp[tick + 1], probability)
        self.model.nodes.newly_infectious[tick] += newly_infectious
        exp[tick + 1] -= newly_infectious
        inf[tick + 1] += newly_infectious

        return

    def end_step(self, tick: int) -> None:
        pass

    @property
    def properties(self) -> list[PropertyType]:
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
            ("newly_infectious", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list:
        return ["E", "I"]


class Infectious:
    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        self.validating = validating

        return

    def setup(self) -> None:
        self.model.states.I[0] = self.model.scenario.I
        return

    def start_step(self, tick: int) -> None:
        inf = self.model.states.I
        inf[tick + 1] = inf[tick]
        return

    def step(self, tick: int) -> None:
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        inf = self.model.states.I
        mortality = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        inf[tick + 1] -= mortality

        return

    def end_step(self, tick: int) -> None:
        pass

    @property
    def properties(self) -> list[PropertyType]:
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return ["I"]


class InfectiousToRecovered(Infectious):
    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        super().__init__(model, mu, validating)
        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        super().step(tick)  # handles non-disease mortality

        # Disease progression - recovery
        inf = self.model.states.I
        probability = -np.expm1(-self.gamma[tick])
        newly_recovered = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.newly_recovered[tick] += newly_recovered
        inf[tick + 1] -= newly_recovered
        self.model.states.R[tick + 1] += newly_recovered

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        return super().properties + [
            ("newly_recovered", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return super().states + ["R"]


class InfectiousToSusceptible(Infectious):
    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        super().__init__(model, mu, validating)
        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        super().step(tick)  # handles non-disease mortality

        # Disease progression - recovery
        inf = self.model.states.I
        probability = -np.expm1(-self.gamma[tick])
        newly_susceptible = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.newly_susceptible[tick] += newly_susceptible
        inf[tick + 1] -= newly_susceptible
        self.model.states.S[tick + 1] += newly_susceptible

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        return super().properties + [
            ("newly_susceptible", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        # + ["S"] almost certainly unnecessary, but for completeness
        return super().states + ["S"]


class Recovered:
    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        self.validating = validating

        return

    def setup(self) -> None:
        self.model.states.R[0] = self.model.scenario.R
        return

    def start_step(self, tick: int) -> None:
        rec = self.model.states.R
        rec[tick + 1] = rec[tick]
        return

    def step(self, tick: int) -> None:
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        rec = self.model.states.R
        mortality = np.random.binomial(rec[tick + 1], probability).astype(rec.dtype)
        self.model.nodes.non_disease_mortality += mortality
        rec[tick + 1] -= mortality

        return

    def end_step(self, tick: int) -> None:
        pass

    @property
    def properties(self) -> list[PropertyType]:
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return ["R"]


class RecoveredToSusceptible(Recovered):
    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        super().__init__(model, mu, validating)

        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))

        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        super().step(tick)  # handles non-disease mortality

        # Disease progression - waning
        rec = self.model.states.R
        probability = -np.expm1(-self.gamma[tick])
        newly_susceptible = np.random.binomial(rec[tick + 1], probability).astype(rec.dtype)
        self.model.nodes.newly_susceptible[tick] += newly_susceptible
        rec[tick + 1] -= newly_susceptible
        self.model.states.S[tick + 1] += newly_susceptible

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        return super().properties + [
            ("newly_susceptible", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        # + ["S"] almost certainly unnecessary, but for completeness
        return super().states + ["S"]


class TransmissionCommon:
    def __init__(self, model, beta: ValuesMap, sink_name: str, flow_name: str, validating: bool = False) -> None:
        self.model = model
        self.beta = beta
        self.validating = validating

        self.sink_name = sink_name
        self.flow_name = flow_name

        return

    def setup(self) -> None:
        self._sink = getattr(self.model.states, self.sink_name)
        self._flow = getattr(self.model.nodes, self.flow_name)
        return

    def start_step(self, tick: int) -> None:
        pass

    def step(self, tick: int) -> None:
        S = self.model.states.S[tick + 1]
        I = self.model.states.I[tick + 1]  # noqa: E741
        N = self.model.states[tick + 1].sum(axis=self.model.states.state_axis - 1)
        rates = self.beta[tick] * I / N
        probabilities = -np.expm1(-rates)
        newly_infected = np.random.binomial(S, probabilities).astype(self._flow.dtype)
        self._flow[tick] = newly_infected
        S -= newly_infected
        self._sink[tick + 1] += newly_infected
        return

    def end_step(self, tick: int) -> None:
        pass

    @property
    def properties(self) -> list[PropertyType]:
        return []

    @property
    def states(self) -> list[str]:
        return ["S"]


class TransmissionSI(TransmissionCommon):
    def __init__(self, model, beta: ValuesMap, validating: bool = False) -> None:
        super().__init__(model, beta, "I", "newly_infectious", validating)
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     super().start_step(tick)
    #     return

    # def step(self, tick: int) -> None:
    #     super().step(tick)
    #     return

    # def end_step(self, tick: int) -> None:
    #     supert().end_step(tick)
    #     return

    @property
    def properties(self) -> list[PropertyType]:
        return super().properties + [
            ("newly_infectious", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return super().states + ["I"]


class TransmissionSE(TransmissionCommon):
    def __init__(self, model, beta: ValuesMap, validating: bool = False) -> None:
        super().__init__(model, beta, "E", "newly_infected", validating)

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     super().start_step(tick)
    #     return

    # def step(self, tick: int) -> None:
    #     super().step(tick)
    #     return

    # def end_step(self, tick: int) -> None:
    #     supert().end_step(tick)
    #     return

    @property
    def properties(self) -> list:
        return super().properties + [
            ("newly_infected", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        return super().states + ["E"]
