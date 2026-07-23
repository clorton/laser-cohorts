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
    """Given a model with S=1000 and r_mortality=0, when the model runs for 10 ticks,
    then S is unchanged at every tick — zero mortality rate produces zero deaths.
    """
    scenario = _scenario(s_init=1000)
    params = PropertySet({"nticks": 10})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=0),
    ]
    model.run()
    assert np.all(model.states.S == 1000)


def test_certain_mortality_empties_compartment() -> None:
    """Given a model with S=500 and r_mortality=inf, when the model runs for 1 tick,
    then S is 0 — r_mortality=inf converts to probability=1, guaranteeing all deaths.
    """
    scenario = _scenario(s_init=500)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=np.inf),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)


def test_deaths_accumulated_in_node_property() -> None:
    """Given a model with S=800 and r_mortality=inf, when the model runs for 1 tick,
    then non_disease_mortality[0] equals 800 — every death is recorded.

    Failure implies the node property accumulation is broken or off-by-one.
    """
    scenario = _scenario(s_init=800)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=np.inf),
    ]
    model.run()
    assert np.all(model.nodes.non_disease_mortality[0] == 800)


def test_subset_states_leaves_other_states_unchanged() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    r_mortality=inf and states={'S'} (a set), then S becomes 0 while I remains 300.

    Failure implies the state mask is not restricting mortality correctly.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states={"S"}),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 300)


def test_states_as_list_restricts_mortality() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    r_mortality=inf and states=['S'] (a list), then S becomes 0 while I remains 300.

    Failure implies list iterables are not accepted or not handled correctly.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states=["S"]),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 300)


def test_states_as_tuple_restricts_mortality() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    r_mortality=inf and states=('S',) (a tuple), then S becomes 0 while I remains 300.

    Failure implies tuple iterables are not accepted or not handled correctly.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states=("S",)),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 300)


def test_all_states_affected_when_states_is_none() -> None:
    """Given a model with S=500 and I=300, when NonDiseaseMortality runs with
    r_mortality=inf and states=None, then both S and I become 0.

    Failure implies the default (all-states) behaviour is not working.
    """
    scenario = _scenario(s_init=500, i_init=300)
    params = PropertySet({"nticks": 1})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        Infectious(model),
        NonDiseaseMortality(model, r_mortality=np.inf, states=None),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 0)
    assert np.all(model.states.I[-1] == 0)


def test_scalar_r_mortality_is_converted_to_valuesmap() -> None:
    """Given a scalar r_mortality=0.05, when NonDiseaseMortality is constructed,
    then its r_mortality attribute is a ValuesMap — scalar inputs must always be
    normalised.
    """
    scenario = _scenario()
    params = PropertySet({"nticks": 5})
    model = Model(scenario, params)
    ndm = NonDiseaseMortality(model, r_mortality=0.05)
    assert isinstance(ndm.r_mortality, ValuesMap)


def test_valuesmap_r_mortality_accepted_unchanged() -> None:
    """Given r_mortality as a zero-valued ValuesMap, when the model runs for 5 ticks,
    then S is unchanged — ValuesMap inputs are used directly without conversion.
    """
    scenario = _scenario(s_init=600)
    nticks = 5
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params)
    r_mortality = ValuesMap.from_scalar(0, nticks, len(scenario))
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=r_mortality),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 600)


def test_ndarray_r_mortality_accepted_unchanged() -> None:
    """Given r_mortality as a 2-D zero numpy array of shape (nticks, nnodes), when
    the model runs for 5 ticks, then S is unchanged — ndarray inputs are used directly.
    """
    scenario = _scenario(s_init=400)
    nticks = 5
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params)
    r_mortality = np.zeros((nticks, len(scenario)))
    model.components = [
        Susceptible(model),
        NonDiseaseMortality(model, r_mortality=r_mortality),
    ]
    model.run()
    assert np.all(model.states.S[-1] == 400)


# ---------------------------------------------------------------------------
# __init__: validation — type and shape of r_mortality
# ---------------------------------------------------------------------------


def test_rejects_non_scalar_non_valuesmap_non_ndarray_r_mortality() -> None:
    """Given a `r_mortality` argument that is neither scalar nor `ValuesMap`
    nor `np.ndarray` (e.g. a nested Python list), when `NonDiseaseMortality`
    is constructed, then a `ValueError` is raised that names the offending
    type.

    The constructor's type-guard should reject anything that can't be
    indexed as ``r_mortality[tick]`` so the failure surfaces at construction
    rather than confusingly inside `step()`.
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    bad_input = [[0.01, 0.01]] * nticks  # nested list, not an ndarray
    with pytest.raises(ValueError, match="must be a scalar"):
        NonDiseaseMortality(model, r_mortality=bad_input)


def test_rejects_dict_r_mortality() -> None:
    """Given a dict passed as `r_mortality`, when `NonDiseaseMortality` is
    constructed, then a `ValueError` is raised.

    Catches the case where a caller passes config-style mappings (e.g.
    keyed by node name) instead of an array.
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    with pytest.raises(ValueError, match="must be a scalar"):
        NonDiseaseMortality(model, r_mortality={"node0": 0.01, "node1": 0.02})


def test_rejects_r_mortality_ndarray_with_wrong_nticks() -> None:
    """Given an ndarray with too many ticks (shape (nticks+1, nnodes)), when
    `NonDiseaseMortality` is constructed, then a `ValueError` is raised that
    mentions ``shape``.

    Catches the off-by-one mistake of sizing the rate array to
    ``nticks + 1`` (the state-array tick dimension) instead of ``nticks``
    (the rate-array tick dimension).
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.full((nticks + 1, 2), 0.01)
    with pytest.raises(ValueError, match="shape"):
        NonDiseaseMortality(model, r_mortality=wrong)


def test_rejects_r_mortality_ndarray_with_wrong_nnodes() -> None:
    """Given an ndarray with the wrong node count (shape (nticks, nnodes+1)),
    when `NonDiseaseMortality` is constructed, then a `ValueError` is raised.

    Catches mis-aligned per-node mortality vectors — e.g. extending the
    network without re-sizing the rate array.
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.full((nticks, 3), 0.01)
    with pytest.raises(ValueError, match="shape"):
        NonDiseaseMortality(model, r_mortality=wrong)


def test_rejects_r_mortality_1d_ndarray() -> None:
    """Given a 1-D ndarray (length nnodes) passed as `r_mortality`, when
    `NonDiseaseMortality` is constructed, then a `ValueError` is raised.

    A length-nnodes vector is a tempting but unsupported shorthand for
    "per-node, constant in time" — should fail loudly with a shape error.
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.array([0.01, 0.02])  # 1-D, length nnodes
    with pytest.raises(ValueError, match="shape"):
        NonDiseaseMortality(model, r_mortality=wrong)


def test_rejects_r_mortality_valuesmap_with_wrong_shape() -> None:
    """Given a `ValuesMap` whose shape doesn't match ``(nticks, nnodes)``,
    when `NonDiseaseMortality` is constructed, then a `ValueError` is
    raised that mentions ``shape``.

    Lets users assert that a pre-built ValuesMap really matches the model
    dimensions before it's used by `step()`.
    """
    import pytest

    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    vmap = ValuesMap.from_array(np.full((nticks, 5), 0.01))  # 5 nodes ≠ 2
    with pytest.raises(ValueError, match="shape"):
        NonDiseaseMortality(model, r_mortality=vmap)


def test_accepts_r_mortality_valuesmap_with_matching_shape() -> None:
    """Given a `ValuesMap.from_array(...)` of the correct shape, when
    `NonDiseaseMortality` is constructed, then the ValuesMap is stored
    verbatim and `r_mortality[tick]` returns the expected per-node vector.

    Confirms the validation gate does not also wrap a well-formed ValuesMap
    into another ValuesMap or otherwise mutate it.
    """
    scenario = _scenario(n_nodes=2)
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    raw = np.full((nticks, 2), 0.005)
    vmap = ValuesMap.from_array(raw)
    comp = NonDiseaseMortality(model, r_mortality=vmap)

    assert comp.r_mortality is vmap
    assert np.allclose(np.asarray(comp.r_mortality[0]), 0.005)
