import matplotlib.pyplot as plt
import numpy as np
from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SI as SI


def run_model(interactive: bool = False) -> Model:
    """Build and run a 9-node SI model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes an SI model (no recovery) with beta=1/30.

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
        }
    )
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))

    components = [
        SI.Susceptible(model),
        SI.Infectious(model),
        SI.Transmission(model, beta=betas),
    ]

    model.components = components

    model.run()

    if interactive:
        plt.plot(model.states.S.sum(axis=1), "blue", label="Susceptible")
        plt.plot(model.states.I.sum(axis=1), "red", label="Infectious")
        plt.plot(model.nodes.newly_infectious.sum(axis=1).cumsum(), ".", label="incidence")  # sum across time
        plt.title("SI")
        plt.grid()
        plt.legend()
        plt.show()

    return model


def test_si():
    """Given a 9-node SI model (no recovery), when the model runs for 5 years,
    then the susceptible compartment is fully depleted and all individuals are
    infectious.
    """
    model = run_model(interactive=False)
    assert np.all(model.states.S[-1] == 0)
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    assert np.all(model.states.I[-1] == N)  # eventually everyone is infected/infectious

    return


if __name__ == "__main__":
    run_model(interactive=True)
