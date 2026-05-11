"""Tests for NonDiseaseMortality in vitaldynamics.py."""

import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model, Susceptible, Infectious
from laser.cohorts.vitaldynamics import NonDiseaseMortality
from laser.generic.utils import ValuesMap


def _scenario(n_nodes: int = 1, s_init: int = 1000, i_init: int = 0):
    """Return a minimal scenario with controlled initial populations."""
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = s_init
    scenario["I"] = i_init
    return scenario


def test_zero_mortality_preserves_populations() -> None:
    """Given a model with S=1000 and mu=0, when the model runs for 10 ticks,
    then S is unchanged at every tick — zero mortality rate produces zero deaths.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 10})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, mu=0),
    ]
    model.run()
    assert np.all(model.states.S == 1000)


def test_certain_mortality_empties_compartment() -> None:
    """Given a model with S=500 and mu=inf, when the model runs for 1 tick,
    then S is 0 — mu=inf converts to probability=1, guaranteeing all deaths.
    """
    scenario = _scenario(s_init=500)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, mu=np.inf),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)


def test_deaths_accumulated_in_node_property() -> None:
    """Given a model with S=800 and mu=inf, when the model runs for 1 tick,
    then non_disease_mortality[0] equals 800 — every death is recorded.

    Failure implies the node property accumulation is broken or off-by-one.
    """
    scenario = _scenario(s_init=800)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, mu=np.inf),
    ]
    model.run()
    assert np.all(model.nodes.non_disease_mortality[0] == 800)


def test_subset_states_leaves_other_states_unchanged() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    mu=inf and states={'S'}, then S becomes 0 while I remains 300.

    Failure implies the state mask is not restricting mortality correctly.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, mu=np.inf, states={"S"}),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 300)


def test_all_states_affected_when_states_is_none() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    mu=inf and states=None, then both S and I become 0.

    Failure implies the default (all-states) behaviour is not working.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, mu=np.inf, states=None),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 0)


def test_scalar_mu_is_converted_to_valuesmap() -> None:
    """Given a scalar mu=0.05, when NonDiseaseMortality is constructed, then its
    mu attribute is a ValuesMap — scalar inputs must always be normalised.
    """
    scenario = _scenario()
    params = PropertySet({"nticks": 5})
    model = Model(scenario, params)
    ndm = NonDiseaseMortality(model, mu=0.05)
    assert isinstance(ndm.mu, ValuesMap)


def test_valuesmap_mu_accepted_unchanged() -> None:
    """Given mu as a zero-valued ValuesMap, when the model runs for 5 ticks,
    then S is unchanged — ValuesMap inputs are used directly without conversion.
    """
    scenario = _scenario(s_init=600)
    nticks = 5
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params)
    mu = ValuesMap.from_scalar(0, nticks, len(scenario))
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, mu=mu),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 600)


def test_ndarray_mu_accepted_unchanged() -> None:
    """Given mu as a 2-D zero numpy array of shape (nticks, nnodes), when the
    model runs for 5 ticks, then S is unchanged — ndarray inputs are used directly.
    """
    scenario = _scenario(s_init=400)
    nticks = 5
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params)
    mu_array = np.zeros((nticks, len(scenario)))
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, mu=mu_array),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 400)
