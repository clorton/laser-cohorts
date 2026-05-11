"""Integration test for the SEI model preset."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SEI as SEI


def run_model(interactive: bool = False, params: dict | None = None) -> Model:
    """Build and run a 9-node SEI model for 5 years.

    Constructs a 3×3 grid scenario, seeds 1% of each node's population as
    infectious (minimum 25, capped at node population),
    and executes an SEI model (no recovery) with beta=1/30 and r_progression=1/7.

    Args:
        interactive (bool): If True, display a matplotlib plot of compartment
            trajectories.
        params (dict | None): Optional parameter overrides. Keys may include
            ``"nticks"``, ``"beta"``, and ``"r_progression"``. Missing keys use the
            default values.

    Returns:
        Model: The completed model instance after all ticks have run.
    """
    scenario = grid(M=3, N=3)
    seeds = np.maximum(np.minimum(25, scenario.S.values), (scenario.S.values * 0.01).astype(int))
    scenario["S"] -= seeds
    scenario["I"] += seeds
    p = PropertySet({
        "nticks": 5 * 365,
        "beta": 1.0 / 30.0,  # 1 new infection per existing infection every 30 ticks
        "r_progression": 1.0 / 7.0,  # 7 ticks of incubation (exposure)
        **(params or {}),
    })
    model = Model(scenario, p)

    betas = ValuesMap.from_scalar(p.beta, p.nticks, len(scenario))  # ty: ignore unresolved-attribute
    r_progression = ValuesMap.from_scalar(p.r_progression, p.nticks, len(scenario))  # ty: ignore unresolved-attribute

    components = [
        SEI.Susceptible(model),
        SEI.Exposed(model, r_progression=r_progression),
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


def test_sei() -> None:
    """Given a 9-node SEI model with no recovery mechanism, when the model runs for
    5 years, then the entire initial population has accumulated in the infectious
    compartment.
    """
    model = run_model(params={"nticks": 5 * 365, "beta": 1.0 / 30.0, "r_progression": 1.0 / 7.0})
    t0 = model.states[0]
    # use state_axis - 1 since taking the last tick reduces dimensionality by 1
    axis = model.states.state_axis - 1
    initial_pops = t0.sum(axis=axis)
    assert np.all(model.states.I[-1] == initial_pops)  # everyone ends up infected/infectious

    return


if __name__ == "__main__":
    def _parse_value(s: str) -> int | float:
        try:
            return int(s)
        except ValueError:
            return float(s)

    parser = argparse.ArgumentParser(description="Run the SEI model.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("params", nargs="*", metavar="KEY=VALUE", help="Parameter overrides, e.g. beta=0.2 nticks=365")
    args = parser.parse_args()

    overrides: dict = {}
    for item in args.params:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)

    run_model(interactive=args.interactive, params=overrides or None)
