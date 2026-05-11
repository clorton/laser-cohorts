import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SEI as SEI


def run_model(interactive: bool = False) -> Model:
    """Build and run a 9-node SEI model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes an SEI model (no recovery) with beta=1/30 and sigma=1/7.

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
            "beta": 1.0 / 30.0,  # 1 new infection per existing infection every 30 ticks
            "sigma": 1.0 / 7.0,  # 7 ticks of incubation (exposure)
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))  # ty: ignore unresolved-attribute
    sigmas = ValuesMap.from_scalar(params.sigma, params.nticks, len(scenario))  # ty: ignore unresolved-attribute

    components = [
        SEI.Susceptible(model),
        SEI.Exposed(model, sigma=sigmas),
        SEI.Infectious(model),
        SEI.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")
        plt.plot(model.states.E.sum(axis=1), "orange", label="Exposed")
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.nodes.newly_infected.sum(axis=1).cumsum(), ".", label="infected")  # sum across time
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.title("SEI")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_sei():
    """Given a 9-node SEI model with no recovery mechanism, when the model runs for
    5 years, then the entire initial population has accumulated in the infectious
    compartment.
    """
    model = run_model(interactive=False)
    t0 = model.states[0]
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    axis = model.states.state_axis - 1
    initial_pops = t0.sum(axis=axis)
    assert np.all(model.states.I[-1] == initial_pops)  # everyone ends up infected/infectious

    return


if __name__ == "__main__":
    run_model(interactive=True)
