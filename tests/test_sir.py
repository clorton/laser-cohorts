"""Integration test for the SIR model preset."""

import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SIR as SIR


def run_model(interactive: bool = False) -> Model:
    """Build and run a 9-node SIR model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes a standard SIR model with beta=1.5/7 and gamma=1/7.

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
            "gamma": 1.0 / 7.0,  # 7 ticks to recovery
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))

    components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, gamma=gammas),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    print(model.states[0])

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")  # sum across nodes
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.states.R.sum(axis=1), "green", label="Recovered")
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.plot(model.nodes.newly_recovered.sum(axis=1).cumsum(), ".", label="recoveries")
        plt.title("SIR")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_sir():
    """Given a 9-node SIR model with standard parameters, when the model runs for
    5 years, then S and R each represent approximately 50% of the total population
    (within 10% relative tolerance) and no individuals remain infectious.
    """
    model = run_model(interactive=False)
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    assert np.allclose(model.states.S[-1] / N, 0.5, rtol=0.10)
    assert np.all(model.states.I[-1] == 0)
    assert np.allclose(model.states.R[-1] / N, 0.5, rtol=0.10)

    return


if __name__ == "__main__":
    run_model(interactive=True)
