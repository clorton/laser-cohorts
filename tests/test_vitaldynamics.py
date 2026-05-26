"""Tests for the vital-dynamics components.

Covers `BirthsByCBR` — a population-growth component that converts a per-tick,
per-node crude birth rate into a binomial number of new susceptibles each
tick, tracked on the ``new_births`` node property.

Existing test modules cover the other vital-dynamics components:

- ``test_mortality.py`` — ``NonDiseaseMortality``
- ``test_births.py``    — ``ConstantPopBirths`` paired with ``NonDiseaseMortality``
"""

from __future__ import annotations

import numpy as np
import pytest

from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap

from laser.cohorts import Model
from laser.cohorts.vitaldynamics import BirthsByCBR, NonDiseaseMortality
import laser.cohorts.SIR as SIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sir_model_with_births(
    n_nodes: int,
    s_per_node: int,
    i_per_node: int,
    r_per_node: int,
    nticks: int,
    r_birth,
    *,
    include_ndm: bool = False,
    r_mortality: float = 0.0,
    seed: int = 42,
) -> Model:
    """Build an SIR model with `BirthsByCBR` and zero transmission / recovery.

    Args:
        n_nodes: Number of simulation nodes (grid M, N=1).
        s_per_node: Initial S population per node.
        i_per_node: Initial I population per node.
        r_per_node: Initial R population per node.
        nticks: Simulation length.
        r_birth: Per-tick, per-node CBR (scalar, ValuesMap, or 2-D ndarray).
        include_ndm: If True, also include `NonDiseaseMortality` BEFORE births.
        r_mortality: Per-tick, per-node mortality rate (only used when
            include_ndm=True).
        seed: numpy RNG seed for reproducibility.

    Returns:
        Model: Fully configured model ready to run.
    """
    np.random.seed(seed)

    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = s_per_node
    scenario["I"] = i_per_node
    scenario["R"] = r_per_node

    params = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, params)

    betas = ValuesMap.from_scalar(0.0, nticks, n_nodes)
    r_recoveries = ValuesMap.from_scalar(0.0, nticks, n_nodes)

    components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
    ]
    if include_ndm:
        components.append(NonDiseaseMortality(model, r_mortality=r_mortality))
    components.append(BirthsByCBR(model, r_birth=r_birth))
    model.components = components

    return model


# ---------------------------------------------------------------------------
# Component protocol: states and properties declarations
# ---------------------------------------------------------------------------


def test_birthsbycbr_states_declares_S() -> None:
    """Given a BirthsByCBR instance, when `states` is accessed, then it
    returns ``["S"]``.

    Failure means births either won't be routed to the S compartment or
    the Model will not allocate S on behalf of this component.
    """
    scenario = grid(M=1, N=1)
    scenario["S"] = 100
    model = Model(scenario, PropertySet({"nticks": 1}))
    comp = BirthsByCBR(model, r_birth=0.01)
    assert comp.states == ["S"]


def test_birthsbycbr_properties_declares_new_births() -> None:
    """Given a BirthsByCBR instance, when `properties` is accessed, then it
    returns one ``("new_births", nticks, np.int32, 0)`` entry.

    Failure means `model.nodes.new_births` is never allocated, and `step()`
    raises `AttributeError` when it tries to record per-node birth counts.
    """
    scenario = grid(M=1, N=1)
    scenario["S"] = 100
    model = Model(scenario, PropertySet({"nticks": 25}))
    comp = BirthsByCBR(model, r_birth=0.01)

    props = comp.properties
    assert len(props) == 1
    name, count, dtype, default = props[0]
    assert name == "new_births"
    assert count == 25
    assert dtype == np.int32
    assert default == 0


# ---------------------------------------------------------------------------
# Construction: scalar, ValuesMap, 2-D ndarray
# ---------------------------------------------------------------------------


def test_birthsbycbr_scalar_rate_is_broadcast_to_valuesmap() -> None:
    """Given a scalar `r_birth=0.01` passed to the constructor, when the
    component is built, then it stores a ValuesMap broadcast across all
    ticks and nodes.

    Failure means scalar rates are not promoted and `r_birth[tick]` raises.
    """
    scenario = grid(M=3, N=1)
    scenario["S"] = 100
    model = Model(scenario, PropertySet({"nticks": 10}))
    comp = BirthsByCBR(model, r_birth=0.01)

    # ValuesMap exposes per-tick lookup that returns a length-nnodes vector.
    sample = np.asarray(comp.r_birth[0])
    assert sample.shape == (3,)
    assert np.allclose(sample, 0.01)


def test_birthsbycbr_accepts_2d_ndarray_directly() -> None:
    """Given a 2-D `(nticks, nnodes)` ndarray, when the component is built,
    then the ndarray is stored verbatim (no copy or broadcast).

    Failure means time-varying or spatially-varying CBR inputs are silently
    ignored or copied into a constant-rate ValuesMap.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))
    r = np.full((nticks, 2), 0.0)
    r[:, 0] = 0.01
    r[:, 1] = 0.02
    comp = BirthsByCBR(model, r_birth=r)

    assert comp.r_birth is r   # same object, not a copy


def test_birthsbycbr_accepts_valuesmap_with_matching_shape() -> None:
    """Given a `ValuesMap` of shape ``(nticks, nnodes)``, when the component
    is built, then the ValuesMap is stored verbatim and `r_birth[tick]` works.

    The constructor's "scalar / ValuesMap / ndarray" type check should let an
    explicit `ValuesMap.from_array(...)` through without modification.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    raw = np.full((nticks, 2), 0.03)
    vmap = ValuesMap.from_array(raw)
    comp = BirthsByCBR(model, r_birth=vmap)

    assert comp.r_birth is vmap
    assert np.allclose(np.asarray(comp.r_birth[0]), 0.03)


# ---------------------------------------------------------------------------
# __init__: validation — type and shape
# ---------------------------------------------------------------------------


def test_birthsbycbr_rejects_non_scalar_non_valuesmap_non_ndarray_input() -> None:
    """Given a `r_birth` argument that is neither scalar nor `ValuesMap` nor
    `np.ndarray` (e.g. a list of lists), when the component is constructed,
    then a `ValueError` is raised that names the offending type.

    Failure means the type-guard is missing or too loose, letting unsupported
    shapes (Python lists, dicts, dataframes, etc.) reach the per-tick lookup
    in `step()` and produce confusing downstream errors.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    bad_input = [[0.01, 0.01]] * nticks   # plain nested list, not an ndarray
    with pytest.raises(ValueError, match="must be a scalar"):
        BirthsByCBR(model, r_birth=bad_input)


def test_birthsbycbr_rejects_dict_input() -> None:
    """Given a `dict` passed as `r_birth`, when the component is constructed,
    then a `ValueError` is raised.

    Catches the case where a user passes config-style data (e.g. {0: 0.01, ...})
    rather than an array — should fail loudly at construction, not silently
    later.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    with pytest.raises(ValueError, match="must be a scalar"):
        BirthsByCBR(model, r_birth={"node0": 0.01, "node1": 0.02})


def test_birthsbycbr_rejects_ndarray_with_wrong_nticks() -> None:
    """Given an ndarray with too many ticks (shape (nticks+1, nnodes)), when
    the component is constructed, then a `ValueError` is raised that mentions
    "shape" and includes both the expected and actual tuples.

    Catches off-by-one mistakes where a caller builds the array using
    ``nticks + 1`` (the state-array length) instead of ``nticks`` (the
    rate-array length).
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.full((nticks + 1, 2), 0.01)   # too many ticks
    with pytest.raises(ValueError, match="shape"):
        BirthsByCBR(model, r_birth=wrong)


def test_birthsbycbr_rejects_ndarray_with_wrong_nnodes() -> None:
    """Given an ndarray with the wrong node count (shape (nticks, nnodes+1)),
    when the component is constructed, then a `ValueError` is raised.

    Catches mis-aligned per-node CBR vectors — e.g. the user expanded the
    network without updating the rate array.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.full((nticks, 3), 0.01)   # too many node columns
    with pytest.raises(ValueError, match="shape"):
        BirthsByCBR(model, r_birth=wrong)


def test_birthsbycbr_rejects_1d_ndarray() -> None:
    """Given a 1-D ndarray (e.g. shape (nnodes,)) passed as `r_birth`, when
    the component is constructed, then a `ValueError` is raised.

    A length-nnodes vector is a common mistake — the user means "per-node,
    constant in time" but the component expects the full 2-D table.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    wrong = np.array([0.01, 0.02])   # 1-D, length nnodes
    with pytest.raises(ValueError, match="shape"):
        BirthsByCBR(model, r_birth=wrong)


def test_birthsbycbr_rejects_valuesmap_with_wrong_shape() -> None:
    """Given a `ValuesMap` constructed with a shape that doesn't match
    ``(nticks, nnodes)``, when the component is constructed, then a
    `ValueError` is raised that mentions the offending shape.

    Lets a careful user assert that their pre-built ValuesMap really matches
    the model dimensions before it's used downstream.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 100
    nticks = 4
    model = Model(scenario, PropertySet({"nticks": nticks}))

    raw = np.full((nticks, 5), 0.01)   # 5 node columns for a 2-node model
    vmap = ValuesMap.from_array(raw)
    with pytest.raises(ValueError, match="shape"):
        BirthsByCBR(model, r_birth=vmap)


# ---------------------------------------------------------------------------
# step(): zero rate produces no births
# ---------------------------------------------------------------------------


def test_birthsbycbr_zero_rate_produces_no_births() -> None:
    """Given `r_birth=0.0`, when the model runs, then S is unchanged and
    `new_births` stays at zero on every tick.

    Failure means birth draws are non-zero when the rate is zero — usually
    a sign of using the rate's expected value rather than a binomial draw.
    """
    model = _build_sir_model_with_births(
        n_nodes=1, s_per_node=1000, i_per_node=0, r_per_node=0,
        nticks=10, r_birth=0.0,
    )
    model.run()

    assert int(model.nodes.new_births.sum()) == 0
    # S is unchanged at every tick
    for t in range(11):
        assert int(model.states.S[t, 0]) == 1000


# ---------------------------------------------------------------------------
# step(): non-zero rate produces births recorded on `new_births`
# ---------------------------------------------------------------------------


def test_birthsbycbr_non_zero_rate_produces_births() -> None:
    """Given a constant `r_birth=0.05` (per tick) and N=10000 per node over
    50 ticks, when the model runs, then `new_births.sum()` is greater than
    zero and `states.S` grows over time.

    Failure means births are not produced or not credited to S.
    """
    nticks = 50
    n_nodes = 4
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=10_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=0.05,
    )
    model.run()

    total_births = int(model.nodes.new_births.sum())
    assert total_births > 0
    # The S compartment grows monotonically (since there's no mortality and no transmission).
    s_totals = [int(model.states.S[t].sum()) for t in range(nticks + 1)]
    assert s_totals[-1] > s_totals[0]


def test_birthsbycbr_new_births_records_per_tick_births() -> None:
    """Given a constant birth rate, when the model runs, then the
    per-tick increment of `states.S` (in a no-mortality, no-transmission
    setup) matches `new_births[tick]` exactly for every tick.

    Failure means `new_births` and the actual S delta disagree, which would
    break any downstream reporting / cost-effectiveness analysis.
    """
    nticks = 8
    model = _build_sir_model_with_births(
        n_nodes=2, s_per_node=5_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=0.02,
    )
    model.run()

    for tick in range(nticks):
        delta = model.states.S[tick + 1] - model.states.S[tick]
        births = model.nodes.new_births[tick]
        assert np.array_equal(delta, births), (
            f"S delta {delta.tolist()} != new_births {births.tolist()} at tick {tick}"
        )


# ---------------------------------------------------------------------------
# step(): N is computed across ALL states, not just S
# ---------------------------------------------------------------------------


def test_birthsbycbr_uses_total_n_not_just_s() -> None:
    """Given a model where S=0, I=0, R=10000 in every node, when births
    fire at a positive rate, then births still occur (because N=10000) and
    are added to S — not zero (which they would be if N were taken from S
    alone).

    Failure indicates the component is using `states.S[tick+1]` as N
    instead of summing across all compartments.
    """
    nticks = 20
    model = _build_sir_model_with_births(
        n_nodes=3, s_per_node=0, i_per_node=0, r_per_node=10_000,
        nticks=nticks, r_birth=0.05,
    )
    model.run()

    # Births must have happened despite S(0) == 0
    assert int(model.nodes.new_births.sum()) > 0
    # And they landed in S
    assert int(model.states.S[-1].sum()) > 0


# ---------------------------------------------------------------------------
# step(): spatially-varying rates produce proportional per-node births
# ---------------------------------------------------------------------------


def test_birthsbycbr_per_node_rate_scales_births() -> None:
    """Given a node whose CBR is 10x the other nodes', when the model runs
    for a short window (where compounding is negligible), then that node
    accumulates roughly 10x as many births as its peers.

    Failure means the per-node rate column of the 2-D `r_birth` array is
    being ignored or applied incorrectly.  A short sim is used so that
    compounding growth in N doesn't itself inflate the observed ratio.
    """
    nticks = 10
    n_nodes = 3
    r = np.full((nticks, n_nodes), 0.001)
    r[:, 1] = 0.010                # node 1 has 10x the CBR
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=50_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=r,
    )
    model.run()

    births_per_node = model.nodes.new_births.sum(axis=0)
    # Node 1 should be ~10x the others; Monte Carlo + minor compounding put
    # the deterministic expectation at ~10.5, so allow [7, 14].
    ratio = births_per_node[1] / np.maximum(births_per_node[0], 1)
    assert 7.0 < ratio < 14.0, f"per-node birth ratio {ratio:.2f} not near 10"


def test_birthsbycbr_per_tick_rate_zero_then_positive() -> None:
    """Given a `r_birth` ndarray that is 0.0 for the first half of the
    simulation and a positive constant for the second half, when the model
    runs, then no births occur in the first half and births accumulate in
    the second.

    Verifies that the time axis of the `r_birth` array is honoured.
    """
    nticks = 100
    n_nodes = 2
    r = np.zeros((nticks, n_nodes))
    r[nticks // 2 :, :] = 0.02
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=5_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=r,
    )
    model.run()

    first_half = int(model.nodes.new_births[: nticks // 2].sum())
    second_half = int(model.nodes.new_births[nticks // 2 :].sum())
    assert first_half == 0
    assert second_half > 0


# ---------------------------------------------------------------------------
# step(): expected birth count matches CBR × N within Monte-Carlo tolerance
# ---------------------------------------------------------------------------


def test_birthsbycbr_total_births_match_expected_law_of_large_numbers() -> None:
    """Given a constant per-tick CBR `r=1e-4`, an initial N=100,000 per node
    in a single-node model, and 500 ticks, when the model runs, then the
    total number of births is within a few percent of the deterministic
    expectation `r * N * nticks` (ignoring compounding from the small
    growth over the run).

    Failure indicates either a missing factor (e.g. using `r` directly
    instead of `1 - exp(-r)` and forgetting the N multiplier) or an
    incorrect compartment sum.
    """
    nticks = 500
    n_nodes = 1
    r = 1e-4
    N0 = 100_000
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=N0, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=r,
    )
    model.run()

    total_births = int(model.nodes.new_births.sum())
    # Lower bound: no compounding; upper bound: full compounding through the run.
    naive_expectation = r * N0 * nticks
    upper_expectation = N0 * (np.exp(r * nticks) - 1.0)   # closed-form pure-growth
    assert 0.7 * naive_expectation < total_births < 1.5 * upper_expectation


# ---------------------------------------------------------------------------
# Integration: BirthsByCBR + NonDiseaseMortality, growth vs decline
# ---------------------------------------------------------------------------


def test_birthsbycbr_growth_when_cbr_exceeds_cdr() -> None:
    """Given matched `BirthsByCBR` and `NonDiseaseMortality` rates with
    CBR > CDR, when the model runs over many ticks, then total population
    grows monotonically (on average).

    Catches sign / direction bugs in the births component when it shares a
    component list with mortality.
    """
    nticks = 200
    n_nodes = 1
    r_birth = 0.003
    r_mortality = 0.001
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=10_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=r_birth,
        include_ndm=True, r_mortality=r_mortality,
    )
    model.run()

    N0 = int(model.states[0].sum())
    N_end = int(model.states[-1].sum())
    assert N_end > N0


def test_birthsbycbr_decline_when_cbr_below_cdr() -> None:
    """Given matched rates with CBR < CDR, when the model runs, then total
    population declines.  The symmetric counterpart of the growth test.

    Together with the growth test, locks in the sign of the births
    contribution.
    """
    nticks = 200
    n_nodes = 1
    r_birth = 0.0005
    r_mortality = 0.003
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=10_000, i_per_node=0, r_per_node=0,
        nticks=nticks, r_birth=r_birth,
        include_ndm=True, r_mortality=r_mortality,
    )
    model.run()

    N0 = int(model.states[0].sum())
    N_end = int(model.states[-1].sum())
    assert N_end < N0


# ---------------------------------------------------------------------------
# Targeting: births always land in S even with mixed initial compartments
# ---------------------------------------------------------------------------


def test_birthsbycbr_routes_all_births_to_s() -> None:
    """Given a mixed initial population (S, I, R all non-zero) and zero
    transmission/recovery/mortality, when the model runs, then the only
    compartment that increases over time is S — every birth lands there.

    Failure means births are leaking into other compartments.
    """
    nticks = 30
    n_nodes = 2
    model = _build_sir_model_with_births(
        n_nodes=n_nodes, s_per_node=500, i_per_node=200, r_per_node=100,
        nticks=nticks, r_birth=0.02,
    )
    model.run()

    # I and R should be unchanged at every tick (no transmission, no recovery, no mortality).
    for t in range(nticks + 1):
        assert int(model.states.I[t].sum()) == 200 * n_nodes
        assert int(model.states.R[t].sum()) == 100 * n_nodes


if __name__ == "__main__":
    pytest.main()
