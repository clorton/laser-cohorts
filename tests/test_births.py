"""Tests for ConstantPopBirths in vitaldynamics.py."""

import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model, Susceptible, Infectious
from laser.cohorts.vitaldynamics import ConstantPopBirths, NonDiseaseMortality


def _scenario(n_nodes: int = 1, s_init: int = 1000, i_init: int = 0):
    """Return a minimal scenario with controlled initial populations."""
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = s_init
    scenario["I"] = i_init
    return scenario


def test_births_replenish_deaths_into_s() -> None:
    """Given a model with S=1000, NonDiseaseMortality(r_mortality=inf), and
    ConstantPopBirths, when the model runs for 1 tick, then S[-1] equals 1000.

    Failure implies births are not reading non_disease_mortality correctly or
    are not being added to the S compartment.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=np.inf),
        ConstantPopBirths(model),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 1000)


def test_total_population_conserved_single_state() -> None:
    """Given a model with S=1000 and NonDiseaseMortality(r_mortality=inf) paired
    with ConstantPopBirths, when the model runs for 5 ticks, then the total
    population equals 1000 at every tick.

    Failure implies the birth-death balance is broken over multiple ticks.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 5})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=np.inf),
        ConstantPopBirths(model),
    ]
    model.run()
    total_per_tick = model.states.S.sum(axis=1)  # sum over nodes
    assert np.all(total_per_tick == 1000)


def test_total_population_conserved_multi_state() -> None:
    """Given a model with S=500 and I=300, NonDiseaseMortality(r_mortality=inf,
    states=None), and ConstantPopBirths, when run for 1 tick, then the total
    population (S+I) at tick 1 equals 800 — deaths across all states are
    replaced by births into S.

    Failure implies ConstantPopBirths is not reading total deaths correctly.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states=None),
        ConstantPopBirths(model),
    ]
    model.run()
    total = model.states.S[-1] + model.states.I[-1]
    assert np.all(total == 800)


def test_births_zero_when_no_deaths() -> None:
    """Given a model with S=1000, NonDiseaseMortality(r_mortality=0), and
    ConstantPopBirths, when the model runs for 5 ticks, then S remains 1000
    at every tick — zero deaths produce zero births.

    Failure implies births are not gated on non_disease_mortality being zero.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 5})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=0),
        ConstantPopBirths(model),
    ]
    model.run()
    assert np.all(model.states.S == 1000)


def test_births_only_target_s_not_other_states() -> None:
    """Given a model with S=500 and I=300, NonDiseaseMortality(r_mortality=inf,
    states={'I'}), and ConstantPopBirths, when run for 1 tick, then I[-1]==0
    and S[-1]==800 — births go exclusively to S, not back to I.

    Failure implies births are being added to the wrong compartment.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states={"I"}),
        ConstantPopBirths(model),
    ]
    model.run()
    assert np.all(model.states.I[-1] == 0)
    assert np.all(model.states.S[-1] == 800)


def test_births_without_ndm_adds_zero() -> None:
    """Given a model with S=1000 and only ConstantPopBirths (no NonDiseaseMortality),
    when the model runs for 3 ticks, then S remains 1000 — the non_disease_mortality
    property is zero-initialised so no phantom births appear.

    Failure implies ConstantPopBirths does not register non_disease_mortality
    correctly when used standalone, or reads uninitialised memory.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 3})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        ConstantPopBirths(model),
    ]
    model.run()
    assert np.all(model.states.S == 1000)


def test_births_equal_non_disease_mortality_per_node() -> None:
    """Given a 3-node model with S=1000 per node and NonDiseaseMortality(r_mortality=inf)
    and ConstantPopBirths, when run for 1 tick, then S[-1] per node equals
    non_disease_mortality[0] per node — births exactly match recorded deaths.

    Failure implies per-node death-to-birth accounting is incorrect.
    """
    scenario = _scenario(n_nodes=3, s_init=1000)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=np.inf),
        ConstantPopBirths(model),
    ]
    model.run()
    assert np.all(model.states.S[-1] == model.nodes.non_disease_mortality[0])
