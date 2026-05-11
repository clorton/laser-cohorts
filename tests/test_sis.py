import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SIS as SIS


def run_model(interactive: bool = False) -> Model:
    scenario = grid(M=3, N=3)
    scenario.S -= 10
    scenario.I += 10
    params = PropertySet(
        {
            "nticks": 5 * 365,
            "beta": 1.25 / 7.0,  # 1.25 new infections per existing infection every 7 ticks
            "gamma": 1.0 / 7.0,  # 7 days to recovery
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))

    components = [
        SIS.Susceptible(model),
        SIS.Infectious(model, gamma=gammas),
        SIS.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.plot(model.nodes.newly_susceptible.sum(axis=1).cumsum(), ".", label="waning")
        plt.title("SIS")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_sis():
    model = run_model(interactive=False)
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    assert np.any(model.states.S[-1] < N)  # use any since we might have local elimination
    assert np.any(model.states.I[-1] > 0)  # use any since we might have local elimination

    return


if __name__ == "__main__":
    run_model(interactive=True)
