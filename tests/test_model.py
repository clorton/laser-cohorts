"""Tests for Model carry-forward behaviour in model.py.

The Model carries compartment values from tick t to tick t+1 at the start of
each simulation step.  By default every state is carried forward; passing
``carry_forward_states`` restricts carry-forward to a named subset.
"""

import warnings

import numpy as np

from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model, Susceptible, Infectious


def _model(s_init: int = 1000, i_init: int = 0, nticks: int = 5, **model_kwargs) -> Model:
    """Return a minimal 1-node model with no-op components."""
    scenario = grid(M=1, N=1)
    scenario["S"] = s_init
    scenario["I"] = i_init
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params, **model_kwargs)
    components = [Susceptible(model)]
    if i_init > 0:
        components.append(Infectious(model))
    model.components = components
    return model


def test_default_carry_forward_propagates_all_states() -> None:
    """Given a model with S=1000 and no transitions, when the model runs for 5
    ticks, then S equals 1000 at every tick including the last.

    Failure implies the default carry-forward (all states) is not working.
    """
    model = _model(s_init=1000, nticks=5)
    model.run()
    assert np.all(model.states.S == 1000)


def test_default_carry_forward_propagates_multiple_states() -> None:
    """Given a model with S=500 and I=300 and no transitions, when the model
    runs for 3 ticks, then S==500 and I==300 at every tick.

    Failure implies carry-forward is not applied uniformly to all states.
    """
    model = _model(s_init=500, i_init=300, nticks=3)
    model.run()
    assert np.all(model.states.S == 500)
    assert np.all(model.states.I == 300)


def test_selective_carry_forward_carries_only_named_state() -> None:
    """Given a model with S=500 and I=300 and carry_forward_states=['S'], when
    the model runs for 1 tick, then S[1]==500 but I[1]==0.

    Failure implies selective carry-forward is leaking into unspecified states.
    """
    model = _model(s_init=500, i_init=300, nticks=1, carry_forward_states=["S"])
    model.run()
    assert np.all(model.states.S[-1] == 500)
    assert np.all(model.states.I[-1] == 0)


def test_selective_carry_forward_tuple_form() -> None:
    """Given carry_forward_states=('I',) (a tuple), when the model runs for 1
    tick, then I[1]==300 but S[1]==0.

    Failure implies tuple iterables are not accepted for carry_forward_states.
    """
    model = _model(s_init=500, i_init=300, nticks=1, carry_forward_states=("I",))
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 300)


def test_empty_carry_forward_leaves_all_states_at_zero() -> None:
    """Given carry_forward_states=[] (empty), when the model runs for 1 tick,
    then S[1]==0 — nothing is carried forward.

    Failure implies an empty iterable is not handled correctly and falls back
    to the default all-states behaviour.
    """
    model = _model(s_init=1000, nticks=1, carry_forward_states=[])
    model.run()
    assert np.all(model.states.S[-1] == 0)


def test_carry_forward_happens_before_component_step() -> None:
    """Given S=1000 at tick 0, when the model runs for 1 tick with no state
    mutations in any component, then S[1]==1000 — the carry from tick 0 is
    visible to components' step() calls.

    Failure implies carry-forward runs after step(), not before.
    """
    model = _model(s_init=1000, nticks=1)
    model.run()
    assert np.all(model.states.S[1] == model.states.S[0])


def test_carry_forward_unknown_state_name_emits_warning() -> None:
    """Given carry_forward_states=['S', 'UNKNOWN'], when components are assigned,
    then a UserWarning is emitted for 'UNKNOWN' and S is still carried correctly.

    Failure implies unknown state names are silently ignored rather than flagged,
    making it harder to catch typos in carry_forward_states.
    """
    scenario = grid(M=1, N=1)
    scenario["S"] = 800
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params, carry_forward_states=["S", "UNKNOWN"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.components = [Susceptible(model)]
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "UNKNOWN" in str(caught[0].message)
    model.run()
    assert np.all(model.states.S[-1] == 800)
