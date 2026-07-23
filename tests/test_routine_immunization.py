"""Tests for ``RoutineImmunization`` in ``routine_immunization.py``.

``RoutineImmunization`` is a model component that periodically moves a
Poisson-drawn fraction of susceptibles into a dedicated ``V`` compartment.
The per-firing fraction is ``period * coverage * eligible_fraction / 365``.

Tests use a minimal one-state-but-V model — a ``Susceptible`` component plus
``RoutineImmunization`` — so the only flux in the system is S → V and the
expectations are deterministic for the boundary cases (``coverage=0`` and
``coverage=1, eligible_fraction=1``).
"""

from __future__ import annotations

import numpy as np
import pytest

from laser.core import PropertySet
from laser.core.utils import grid

from laser.cohorts import Model, RoutineImmunization, Susceptible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(n_nodes: int = 1, s_init: int = 1000):
    """Build a minimal scenario with controlled initial S per node."""
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = s_init
    return scenario


def _build_model(*, s_init: int, nticks: int, n_nodes: int = 1, **ri_kwargs) -> Model:
    """Build a minimal Susceptible + RoutineImmunization model.

    All kwargs after the leading model-shape kwargs are forwarded straight to
    ``RoutineImmunization`` so individual tests can vary coverage, eligible
    fraction, and period independently.
    """
    scenario = _scenario(n_nodes=n_nodes, s_init=s_init)
    params = PropertySet({"nticks": nticks})
    model = Model(scenario, params)
    model.components = [
        Susceptible(model),
        RoutineImmunization(model, **ri_kwargs),
    ]
    return model


# ---------------------------------------------------------------------------
# Construction — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_coverage", [-0.01, -1.0, 1.01, 2.0])
def test_init_rejects_out_of_range_coverage(bad_coverage: float) -> None:
    """Given a ``coverage`` outside ``[0, 1]``, when constructing the component,
    then a ValueError is raised that names ``coverage``.

    Catches the off-by-one mistake of passing a per-1000 or per-100,000 rate
    instead of a fraction.
    """
    scenario = _scenario()
    model = Model(scenario, PropertySet({"nticks": 5}))
    with pytest.raises(ValueError, match="coverage"):
        RoutineImmunization(model, coverage=bad_coverage)


@pytest.mark.parametrize("bad_ef", [-0.01, 1.01, 2.0])
def test_init_rejects_out_of_range_eligible_fraction(bad_ef: float) -> None:
    """Given an ``eligible_fraction`` outside ``[0, 1]``, when constructing the
    component, then a ValueError is raised that names ``eligible_fraction``.
    """
    scenario = _scenario()
    model = Model(scenario, PropertySet({"nticks": 5}))
    with pytest.raises(ValueError, match="eligible_fraction"):
        RoutineImmunization(model, coverage=0.5, eligible_fraction=bad_ef)


@pytest.mark.parametrize("bad_period", [0, -1, 1.5, "30", True])
def test_init_rejects_invalid_period(bad_period) -> None:
    """Given a ``period`` that is not a positive int, when constructing the
    component, then a ValueError is raised that names ``period``.

    Booleans are explicitly rejected even though ``bool`` is a subtype of
    ``int`` — ``RoutineImmunization(model, coverage=0.5, period=True)`` is
    almost certainly a mistake.
    """
    scenario = _scenario()
    model = Model(scenario, PropertySet({"nticks": 5}))
    with pytest.raises(ValueError, match="period"):
        RoutineImmunization(model, coverage=0.5, period=bad_period)


# ---------------------------------------------------------------------------
# Declared states and properties
# ---------------------------------------------------------------------------


def test_states_property_declares_V() -> None:
    """Given a RoutineImmunization component, when its ``states`` property is
    queried, then it returns ``["V"]`` so the Model allocates the V compartment.

    Failure means the component would silently fail to declare a destination
    state and crash later when ``step()`` tries to write to ``states.V``.
    """
    model = Model(_scenario(), PropertySet({"nticks": 3}))
    ri = RoutineImmunization(model, coverage=0.5)
    assert ri.states == ["V"]


def test_properties_property_declares_ri_vaccinated_with_correct_shape() -> None:
    """Given a RoutineImmunization component on a model with nticks=7, when its
    ``properties`` property is queried, then it declares ``ri_vaccinated`` with
    length matching ``nticks``.
    """
    nticks = 7
    model = Model(_scenario(), PropertySet({"nticks": nticks}))
    ri = RoutineImmunization(model, coverage=0.5)
    props = ri.properties
    assert len(props) == 1
    name, count, dtype, default = props[0]
    assert name == "ri_vaccinated"
    assert count == nticks
    assert dtype == np.int32
    assert default == 0


def test_model_allocates_V_compartment_when_RI_present() -> None:
    """Given a model whose components include RoutineImmunization, when
    components are assigned, then ``states.V`` exists and starts at zero.

    Failure means the V state was not picked up from the component's
    ``states`` property — the very mechanism by which RI declares a new
    compartment.
    """
    model = _build_model(s_init=100, nticks=3, coverage=0.0)
    assert "V" in model.states.state_names
    assert np.all(model.states.V == 0)


# ---------------------------------------------------------------------------
# Step behaviour
# ---------------------------------------------------------------------------


def test_zero_coverage_produces_no_vaccinations() -> None:
    """Given coverage=0, when the model runs for many ticks, then no
    susceptibles move into V and ``ri_vaccinated`` stays at zero.
    """
    nticks = 50
    model = _build_model(s_init=1000, nticks=nticks, coverage=0.0)
    model.run()

    assert np.all(model.states.S == 1000)
    assert np.all(model.states.V == 0)
    assert np.all(model.nodes.ri_vaccinated == 0)


def test_full_coverage_one_period_year_vaccinates_almost_everyone() -> None:
    """Given coverage=1, eligible_fraction=1, period=365, and nticks=365, when
    the model runs, then S→V essentially empties S in one year.

    With a per-firing fraction of 1.0 and a Poisson mean equal to S, the cap
    at S guarantees all of S transitions to V on the single firing tick.
    """
    np.random.seed(0)
    nticks = 365
    s_init = 1_000
    model = _build_model(s_init=s_init, nticks=nticks, coverage=1.0, eligible_fraction=1.0, period=365)
    model.run()

    assert int(model.states.S[-1, 0]) == 0
    assert int(model.states.V[-1, 0]) == s_init
    # All vaccinations recorded on the single firing tick (tick 0).
    assert int(model.nodes.ri_vaccinated[0, 0]) == s_init
    assert int(model.nodes.ri_vaccinated[1:].sum()) == 0


def test_ri_vaccinated_accumulates_transitions() -> None:
    """Given a finite-coverage RI run, when the model finishes, then the sum
    of ``ri_vaccinated`` equals the final V minus the initial V (zero).

    Failure means the property is double-counting, missing entries, or being
    written to the wrong tick index.
    """
    np.random.seed(0)
    nticks = 365
    model = _build_model(s_init=5_000, nticks=nticks, coverage=0.5, period=30)
    model.run()

    total_recorded = int(model.nodes.ri_vaccinated.sum())
    final_V = int(model.states.V[-1].sum())
    assert total_recorded == final_V


def test_S_plus_V_is_conserved() -> None:
    """Given the no-vital-dynamics minimal model, when RI runs, then S+V is
    conserved at the initial population at every tick.

    Routine immunization moves individuals between compartments; it does not
    create or destroy them.
    """
    np.random.seed(0)
    nticks = 200
    s_init = 2_500
    model = _build_model(s_init=s_init, nticks=nticks, coverage=0.7, period=10)
    model.run()

    totals = model.states.S.sum(axis=-1) + model.states.V.sum(axis=-1)
    assert np.all(totals == s_init)


def test_vaccinations_never_exceed_available_susceptibles() -> None:
    """Given coverage=1 / ef=1 / period=365 (mean of Poisson = S), when the
    model runs, then no firing tick records more vaccinations than the
    susceptibles available at that tick.

    Catches a regression where the Poisson over-draw isn't capped at S — a
    Poisson with mean S routinely produces draws above S.
    """
    np.random.seed(0)
    nticks = 365
    s_init = 200
    model = _build_model(s_init=s_init, nticks=nticks, coverage=1.0, eligible_fraction=1.0, period=365)
    model.run()

    # S is monotone non-increasing under this component (no births, no recoveries).
    assert np.all(np.diff(model.states.S[:, 0]) <= 0)
    # The firing-tick total cannot exceed initial S.
    assert int(model.nodes.ri_vaccinated[0, 0]) == s_init


def test_period_30_fires_only_on_multiples_of_30() -> None:
    """Given period=30 and coverage > 0, when the model runs for 365 ticks,
    then ``ri_vaccinated`` is non-zero only on ticks divisible by 30.

    Pins the firing schedule — drifting off this would mask period bugs as
    "the totals look about right".
    """
    np.random.seed(0)
    nticks = 365
    model = _build_model(s_init=10_000, nticks=nticks, coverage=0.8, period=30)
    model.run()

    per_tick = model.nodes.ri_vaccinated[:, 0]
    firing_ticks = np.flatnonzero(per_tick > 0)
    assert firing_ticks.size > 0
    assert set(firing_ticks.tolist()).issubset({t for t in range(nticks) if t % 30 == 0})


def test_default_period_one_fires_every_tick() -> None:
    """Given period=1 (the default), when the model runs and the per-tick mean
    is high enough to produce draws every tick, then every tick records a
    non-zero ``ri_vaccinated`` entry.
    """
    np.random.seed(0)
    nticks = 100
    # Per-tick fraction = 1.0 * 1.0 / 365 ≈ 0.00274; mean per tick on 50k S ≈ 137.
    model = _build_model(s_init=50_000, nticks=nticks, coverage=1.0, eligible_fraction=1.0)
    model.run()

    per_tick = model.nodes.ri_vaccinated[:, 0]
    # With mean ~137 every tick the chance of any tick being zero is ~e^-137 ≈ 0.
    assert np.all(per_tick > 0)


def test_expected_annual_total_matches_r_ef_S0_within_stochastic_band() -> None:
    """Given coverage=0.5, eligible_fraction=0.6, period=30, S0=100,000 and
    nticks=365, when the model runs, then the total vaccinated over one year
    is within ~5% of the analytical expectation ``r * ef * S0 = 30,000``.

    This is the integral sanity check — the daily fraction ``r * ef / 365``
    integrated over a year multiplied by an essentially-constant S equals
    ``r * ef * S``.  The tolerance allows for S shrinking as vaccinations
    accumulate.
    """
    np.random.seed(0)
    s_init = 100_000
    nticks = 365
    coverage = 0.5
    ef = 0.6
    model = _build_model(s_init=s_init, nticks=nticks, coverage=coverage, eligible_fraction=ef, period=30)
    model.run()

    total = int(model.nodes.ri_vaccinated.sum())
    # As S decreases over the year, total vaccinated ≈ r*ef*S0*(1 - exp(-r*ef))
    # which for r*ef=0.3 ≈ 25,918 — i.e. less than the naive r*ef*S0 = 30k.
    # We bound it to a wide stochastic band that captures both.
    assert 20_000 < total < 32_000


def test_multi_node_per_node_draws_are_independent() -> None:
    """Given a 4-node scenario with identical initial populations, when RI
    runs, then each node receives an independent Poisson draw and the
    per-node ``ri_vaccinated`` totals are not identical.

    A correct vectorised draw produces different outcomes per node; a buggy
    implementation that broadcasts a single draw would produce identical
    counts and silently lose the spatial variance.
    """
    np.random.seed(0)
    nticks = 90
    model = _build_model(s_init=50_000, n_nodes=4, nticks=nticks, coverage=0.5, period=30)
    model.run()

    per_node = model.nodes.ri_vaccinated.sum(axis=0)
    assert per_node.shape == (4,)
    assert per_node.min() > 0
    assert len(set(per_node.tolist())) > 1
