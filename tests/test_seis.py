import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SEIS as SEIS


def run_model(interactive: bool = False) -> Model:
    scenario = grid(M=3, N=3)
    scenario.S -= 10
    scenario.I += 10
    params = PropertySet(
        {
            "nticks": 5 * 365,
            "beta": 1.0 / 30.0,  # 1 new infection per existing infection every 30 ticks
            "sigma": 1.0 / 7.0,  # 7 ticks of incubation (exposure)
            "gamma": 1.0 / 180.0,  # 7 ticks to recovery
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
    sigmas = ValuesMap.from_scalar(params.sigma, params.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))

    components = [
        SEIS.Susceptible(model),
        SEIS.Exposed(model, sigma=sigmas),
        SEIS.Infectious(model, gamma=gammas),
        SEIS.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")
        plt.plot(model.states.E.sum(axis=1), "orange", label="Exposed")
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.nodes.newly_infected.sum(axis=1).cumsum(), ".", label="infected")  # sum across time
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.plot(model.nodes.newly_susceptible.sum(axis=1).cumsum(), ".", label="waning")  # sum across time
        plt.title("SEIS")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_seis():
    model = run_model(interactive=False)
    assert np.all(model.states.S[-1] > 0)
    assert np.all(model.states.S[-1] < model.states.I[-1])
    assert np.all(model.states.E[-1] > 0)
    assert np.all(model.states.E[-1] < model.states.S[-1])
    # implicitly checked above with S < I
    # assert np.all(model.states.I[-1] > model.states.S[-1])

    return


if __name__ == "__main__":
    run_model(interactive=True)
