import numpy as np

class Susceptible:

    def __init__(self, model, validating: bool = False):
        self.model = model
        self.validating = validating

        model.nodes.add_vector_property("S", model.params.nticks + 1, dtype=np.int32)
        model.nodes.S[0] = model.scenario.S

        return

    def step(self, tick: int) -> None:
        return


class Exposed:

    def __init__(self, model, sigma, validating: bool = False):
        self.model = model
        self.sigma = sigma
        self.validating = validating

        model.nodes.add_vector_property("E", model.params.nticks + 1, dtype=np.int32)
        model.nodes.add_vector_property("newly_infectious", model.params.nticks + 1, dtype=np.int32)

        model.nodes.E[0] = model.scenario.E

        return

    def step(self, tick: int) -> None:

        self.model.nodes.newly_infectious[tick] = (newly_infectious := np.binomial(self.model.nodes.E[tick], self.sigma[tick]))
        self.model.nodes.E[tick + 1] += self.model.nodes.E[tick] - newly_infectious

        return


class Infectious:

    def __init__(self, model, gamma, validating: bool = False):
        self.model = model
        self.gamma = gamma
        self.validating = validating

        model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        if np.any(gamma != 0):
            model.nodes.add_vector_property("newly_recovered", model.params.nticks + 1, dtype=np.int32)
            if not hasattr(model.nodes, "R"):
                model.nodes.add_vector_property("R", model.params.nticks + 1, dtype=np.int32)

        model.nodes.I[0] = model.scenario.I

        return

    def step(self, tick: int) -> None:

        newly_recovered = np.binomial(self.model.nodes.I[tick], self.gamma)
        if np.any(newly_recovered != 0):
            self.model.nodes.newly_recovered[tick] = newly_recovered
            self.model.nodes.R[tick+1] += newly_recovered
        self.model.nodes.I[tick+1] += self.model.nodes.I[tick] - newly_recovered

        return

class Recovered:

    def __init__(self, model, validating: bool = False):

        self.model = model
        self.validating = validating

        model.nodes.add_vector_property("R", model.params.nticks + 1, dtype=np.int32)
        model.nodes.R[0] = model.scenario.R

        return