import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SEIRS as SEIRS


def run_model(interactive: bool = False) -> Model:
    scenario = grid(M=3, N=3)
    scenario.S -= 10
    scenario.I += 10
    params = PropertySet(
        {
            "nticks": 5 * 365,
            # "beta": 1.386/7.0, # 1.386 new infections per existing infection every 7 ticks
            "beta": 1.5 / 7.0,
            "sigma": 1.0 / 7.0,  # 7 ticks of incubation (exposure)
            "gamma": 1.0 / 7.0,  # 7 ticks to recovery
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
    sigmas = ValuesMap.from_scalar(params.sigma, params.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))

    components = [
        SEIRS.Susceptible(model),
        SEIRS.Exposed(model, sigma=sigmas),
        SEIRS.Infectious(model, gamma=gammas),
        SEIRS.Recovered(model),
        SEIRS.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    print(model.states[0])

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")  # sum across nodes
        plt.plot(model.states.E.sum(axis=1), "orange", label="Exposed")
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.states.R.sum(axis=1), "green", label="Recovered")
        plt.plot(model.nodes.newly_infected.sum(axis=1).cumsum(), ".", label="infections")  # sum across time
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.plot(model.nodes.newly_recovered.sum(axis=1).cumsum(), ".", label="recoveries")
        plt.plot(model.nodes.newly_susceptible.sum(axis=1).cumsum(), ".", label="waning")
        plt.title("SEIRS")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_seirs():
    model = run_model(interactive=False)
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    assert np.allclose(model.states.S[-1] / N, 0.5, rtol=0.10)
    assert np.all(model.states.E[-1] == 0)
    assert np.all(model.states.I[-1] == 0)
    assert np.allclose(model.states.R[-1] / N, 0.5, rtol=0.10)

    return


if __name__ == "__main__":
    run_model(interactive=True)
