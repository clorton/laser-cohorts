"""Integration test for the SIRS model preset."""

import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SIRS as SIRS


def run_model(interactive: bool = False) -> Model:
    """Build and run a 9-node SIRS model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes an SIRS model with beta=1.5/7, gamma=1/7, and waning=1/30.

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
            "beta": 1.5 / 7.0,  # 1.25 new infections per existing infection every 7 ticks
            "gamma": 1.0 / 7.0,  # 7 days to recovery
            "waning": 1.0 / 30.0,  # 30 days to susceptibility
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))
    waning = ValuesMap.from_scalar(params.waning, params.nticks, len(scenario))

    components = [
        SIRS.Susceptible(model),
        SIRS.Infectious(model, gamma=gammas),
        SIRS.Recovered(model, gamma=waning),
        SIRS.Transmission(model, beta=betas),
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
        plt.title("SIRS")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_sirs():
    """Given a 9-node SIRS model with waning immunity, when the model runs for 5 years,
    then the compartment ordering S > R >= I holds at all nodes, reflecting the waning
    immunity dynamic that recycles recovered individuals back to susceptible.
    """
    model = run_model(interactive=False)
    assert np.all(model.states.S[-1] > model.states.R[-1])
    assert np.all(model.states.I[-1] >= 0)
    assert np.all(model.states.R[-1] >= model.states.I[-1])
    assert np.any(model.states.R[-1] > model.states.I[-1])

    return


if __name__ == "__main__":
    run_model(interactive=True)
