"""Integration test for the SIRS model preset."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import laser.core.random
from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SIRS as SIRS


def run_model(interactive: bool = False, params: dict | None = None) -> Model:
    """Build and run a 9-node SIRS model for 5 years.

    Constructs a 3×3 grid scenario, seeds 10 infectious individuals per node,
    and executes an SIRS model with beta=1.5/7, gamma=1/7, and waning=1/30.

    Args:
        interactive (bool): If True, display a matplotlib plot of compartment
            trajectories.
        params (dict | None): Optional parameter overrides. Keys may include
            ``"nticks"``, ``"beta"``, ``"gamma"``, and ``"waning"``. Missing
            keys use the default values.

    Returns:
        Model: The completed model instance after all ticks have run.
    """
    scenario = grid(M=3, N=3)
    scenario.S -= 10
    scenario.I += 10
    p = PropertySet({
        "nticks": 5 * 365,
        "beta": 1.5 / 7.0,  # 1.25 new infections per existing infection every 7 ticks
        "gamma": 1.0 / 7.0,  # 7 days to recovery
        "omega": 1.0 / 30.0,  # 30 days to susceptibility
        **(params or {}),
    })
    model = Model(scenario, p)

    betas = ValuesMap.from_scalar(p.beta, p.nticks, len(scenario))
    gammas = ValuesMap.from_scalar(p.gamma, p.nticks, len(scenario))
    omegas = ValuesMap.from_scalar(p.omega, p.nticks, len(scenario))

    components = [
        SIRS.Susceptible(model),
        SIRS.Infectious(model, gamma=gammas),
        SIRS.Recovered(model, omega=omegas),
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


def test_sirs() -> None:
    """Given a 9-node SIRS model with waning immunity, when the model runs for 5 years,
    then the compartment ordering S > R >= I holds at all nodes, reflecting the waning
    immunity dynamic that recycles recovered individuals back to susceptible.

    Seed is fixed so that no node experiences stochastic epidemic extinction.
    """
    laser.core.random.seed(0)
    model = run_model(params={"nticks": 5 * 365, "beta": 1.5 / 7.0, "gamma": 1.0 / 7.0, "waning": 1.0 / 30.0})
    assert np.all(model.states.S[-1] > model.states.R[-1])
    assert np.all(model.states.I[-1] >= 0)
    assert np.all(model.states.R[-1] >= model.states.I[-1])
    assert np.any(model.states.R[-1] > model.states.I[-1])

    return


if __name__ == "__main__":
    def _parse_value(s: str) -> int | float:
        try:
            return int(s)
        except ValueError:
            return float(s)

    parser = argparse.ArgumentParser(description="Run the SIRS model.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("params", nargs="*", metavar="KEY=VALUE", help="Parameter overrides, e.g. beta=0.2 nticks=365")
    args = parser.parse_args()

    overrides: dict = {}
    for item in args.params:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)

    run_model(interactive=args.interactive, params=overrides or None)
