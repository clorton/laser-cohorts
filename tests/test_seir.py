"""Integration test for the SEIR model preset."""

import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SEIR as SEIR


def run_model(interactive: bool = False) -> Model:
    """Build and run a 9-node SEIR model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes an SEIR model with beta=1.5/7, sigma=1/7, and gamma=1/7.

    Args:
        interactive (bool): If True, display a matplotlib plot of compartment
            trajectories.

    Returns:
        Model: The completed model instance after all ticks have run.
    """
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
        SEIR.Susceptible(model),
        SEIR.Exposed(model, sigma=sigmas),
        SEIR.Infectious(model, gamma=gammas),
        SEIR.Recovered(model),
        SEIR.Transmission(model, beta=betas),
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
        plt.plot(model.nodes.newly_recovered.sum(axis=1).cumsum(), ".", label="recoveries")  # sum across time
        plt.title("SEIR")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_seir():
    """Given a 9-node SEIR model with standard parameters, when the model runs for
    5 years, then the epidemic exhausts and S and R each represent approximately 50%
    of the total population (within 10% relative tolerance) with no remaining E or I.
    """
    model = run_model(interactive=False)
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    assert np.allclose(model.states.S[-1] / N, 0.5, rtol=0.10)
    assert np.all(model.states.E[-1] == 0)  # no waning immunity, eradication given enough time
    assert np.all(model.states.I[-1] == 0)  # no waning immunity, eradication given enough time
    assert np.allclose(model.states.R[-1] / N, 0.5, rtol=0.10)

    return


if __name__ == "__main__":
    run_model(interactive=True)
