"""Tests for the Vaccination intervention and its integration with Campaign.

Vaccination is a Campaign intervention that moves a binomial fraction of
targeted compartment states into a dedicated V (vaccinated) state.  It
declares the V state and a newly_vaccinated node property so the model
allocates them before the simulation runs.

Test coverage:
- Intervention protocol: states and properties declarations
- Campaign aggregation: Campaign.states/properties surface Vaccination declarations
- Deterministic apply: coverage=0 (no change), coverage=1 (complete transfer)
- Error handling: invalid coverage raises ValueError
- Targeting: who list restricts to named states; who=None targets all states
- Targeting: where list restricts to named nodes; where=None targets all nodes
- Recording: newly_vaccinated property updated correctly
- Conservation: total population invariant across vaccination
- Persistence: vaccinated individuals accumulate in V across ticks
"""

import numpy as np
import pytest

import laser.core.random
from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Campaign, Model
from laser.cohorts.interventions import Vaccination
import laser.cohorts.SIR as SIR
from laser.generic.utils import ValuesMap


Campaign.register(Vaccination)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_model(nticks: int = 5) -> Model:
    """Build a single-node model with params but no components.

    Args:
        nticks (int): Simulation duration in ticks.

    Returns:
        Model: Model with params set but model.states not yet allocated.
    """
    scenario = grid(M=1, N=1)
    scenario["S"] = 100
    scenario["I"] = 0
    scenario["R"] = 0
    return Model(scenario, PropertySet({"nticks": nticks}))


def _build_model(
    n_nodes: int,
    s_per_node: int,
    i_per_node: int,
    nticks: int,
    schedule,
    r_per_node: int = 0,
) -> Model:
    """Build an SIR model with a Campaign containing a Vaccination schedule.

    Uses zero transmission and zero recovery so only the Campaign intervention
    moves individuals between compartments.

    Args:
        n_nodes (int): Number of simulation nodes.
        s_per_node (int): Susceptible population per node.
        i_per_node (int): Infected population per node.
        nticks (int): Simulation duration in ticks.
        schedule: Campaign schedule (dict, list, path, etc.).
        r_per_node (int): Recovered population per node. Defaults to 0.

    Returns:
        Model: Fully configured model ready to run.
    """
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = s_per_node
    scenario["I"] = i_per_node
    scenario["R"] = r_per_node
    params = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, params)
    campaign = Campaign(model, schedule)
    betas = ValuesMap.from_scalar(0.0, nticks, n_nodes)
    r_recoveries = ValuesMap.from_scalar(0.0, nticks, n_nodes)
    model.components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
        campaign,
    ]
    return model


# ---------------------------------------------------------------------------
# Intervention protocol: states and properties declarations
# ---------------------------------------------------------------------------


def test_vaccination_states_declares_V() -> None:
    """Given a Vaccination instance, when states is accessed, then it returns ["V"].

    If this fails, the V compartment will not be allocated by Campaign and any
    attempt to move individuals into V will raise an AttributeError at runtime.
    """
    model = _minimal_model()
    vacc = Vaccination(model)
    assert vacc.states == ["V"]


def test_vaccination_properties_declares_newly_vaccinated() -> None:
    """Given a Vaccination instance, when properties is accessed, then it includes
    a "newly_vaccinated" entry with np.int32 dtype.

    If this fails, model.nodes.newly_vaccinated is never allocated and the
    apply() method raises AttributeError when recording vaccination counts.
    """
    model = _minimal_model(nticks=10)
    vacc = Vaccination(model)
    props = vacc.properties
    assert len(props) == 1
    name, count, dtype, default = props[0]
    assert name == "newly_vaccinated"
    assert count == 10
    assert dtype == np.int32
    assert default == 0


# ---------------------------------------------------------------------------
# Campaign aggregation: surfaces states and properties from interventions
# ---------------------------------------------------------------------------


def test_campaign_states_surfaces_V_from_vaccination() -> None:
    """Given a Campaign scheduled with a Vaccination entry, when campaign.states
    is accessed, then "V" is included in the result.

    Campaign.states must aggregate states from all registered interventions
    referenced in the schedule.  Failure means V is never declared to the Model
    and the vaccination tick raises AttributeError.
    """
    model = _minimal_model()
    schedule = [{"who": "*", "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 0.5}, "notes": ""}]
    campaign = Campaign(model, schedule)
    assert "V" in campaign.states


def test_campaign_properties_surfaces_newly_vaccinated_from_vaccination() -> None:
    """Given a Campaign scheduled with a Vaccination entry, when campaign.properties
    is accessed, then "newly_vaccinated" is included in the result.

    Campaign.properties must aggregate properties from all registered interventions
    referenced in the schedule.  Failure means newly_vaccinated is never allocated
    and the vaccination tick raises AttributeError.
    """
    model = _minimal_model()
    schedule = [{"who": "*", "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 0.5}, "notes": ""}]
    campaign = Campaign(model, schedule)
    names = [p[0] for p in campaign.properties]
    assert "newly_vaccinated" in names


# ---------------------------------------------------------------------------
# Deterministic apply: coverage=0 and coverage=1
# ---------------------------------------------------------------------------


def test_coverage_zero_vaccinates_nobody() -> None:
    """Given a model with 1000 S in one node and a Vaccination at tick 0 with
    coverage=0.0, when the model runs, then S is unchanged and V stays at 0.

    Coverage=0 is the floor for the parameter; zero probability binomial draws
    must produce exactly zero vaccinations.  Failure means a non-zero fraction
    is moved to V even when coverage is explicitly disabled.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 0.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=0, nticks=3, schedule=schedule)
    model.run()

    assert int(model.states.S[1, 0]) == 1000
    assert int(model.states.V[1, 0]) == 0
    assert int(model.nodes.newly_vaccinated[0, 0]) == 0


def test_coverage_one_vaccinates_all_targeted() -> None:
    """Given a model with 1000 S and 200 I in one node and a Vaccination at tick 0
    with coverage=1.0 targeting only S, when the model runs, then all 1000 S move
    to V and I remains at 200.

    Coverage=1.0 is the ceiling; the binomial draw with p=1 is deterministic.
    Failure means not all susceptibles are vaccinated or infected individuals
    are incorrectly moved to V.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=200, nticks=3, schedule=schedule)
    model.run()

    assert int(model.states.S[1, 0]) == 0
    assert int(model.states.V[1, 0]) == 1000
    assert int(model.states.I[1, 0]) == 200


def test_invalid_coverage_raises_value_error() -> None:
    """Given a Vaccination scheduled with coverage=1.5 (out of range), when the
    model runs and reaches that tick, then a ValueError is raised.

    Coverage must be a probability in [0, 1].  Failure means out-of-range
    values are silently clamped or produce undefined statistical behaviour.
    """
    schedule = [{"who": "*", "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.5}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=0, nticks=3, schedule=schedule)
    with pytest.raises(ValueError, match="coverage"):
        model.run()


# ---------------------------------------------------------------------------
# Targeting: who
# ---------------------------------------------------------------------------


def test_who_list_only_vaccinates_named_state() -> None:
    """Given a Vaccination targeting only who=["S"] with coverage=1.0, and a model
    with both S and I, when the model runs, then only S moves to V and I is
    unchanged.

    The who list must restrict vaccination to the named states.  Failure means
    I (or other states) are incorrectly moved to V.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=500, i_per_node=200, nticks=2, schedule=schedule)
    model.run()

    assert int(model.states.S[1, 0]) == 0
    assert int(model.states.I[1, 0]) == 200
    assert int(model.states.V[1, 0]) == 500


def test_who_none_vaccinates_all_states() -> None:
    """Given a Vaccination with who=None (wildcard "*") and coverage=1.0, and a
    model with S=500, I=200, R=100, when the model runs, then all three states
    are fully moved to V.

    None indicates all compartments are targeted.  Failure means only the first
    state or a hardcoded subset is vaccinated rather than every registered state.
    """
    schedule = [{"who": "*", "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=500, i_per_node=200, nticks=2, schedule=schedule, r_per_node=100)
    model.run()

    assert int(model.states.S[1, 0]) == 0
    assert int(model.states.I[1, 0]) == 0
    assert int(model.states.R[1, 0]) == 0
    assert int(model.states.V[1, 0]) == 800


# ---------------------------------------------------------------------------
# Targeting: where
# ---------------------------------------------------------------------------


def test_where_list_only_vaccinates_named_node() -> None:
    """Given a 2-node model with 500 S each and a Vaccination targeting only
    where=[0] with coverage=1.0, when the model runs, then only node 0 is
    fully vaccinated and node 1 is unchanged.

    The where list must restrict vaccination to the named nodes.  Failure means
    node 1 (or all nodes) is incorrectly vaccinated.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": [0], "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=2, s_per_node=500, i_per_node=0, nticks=2, schedule=schedule)
    model.run()

    assert int(model.states.S[1, 0]) == 0
    assert int(model.states.V[1, 0]) == 500
    assert int(model.states.S[1, 1]) == 500
    assert int(model.states.V[1, 1]) == 0


def test_where_none_vaccinates_all_nodes() -> None:
    """Given a 2-node model with 500 S each and a Vaccination targeting where="*"
    with coverage=1.0, when the model runs, then both nodes are fully vaccinated.

    None (from "*") indicates all nodes are targeted.  Failure means only a
    subset of nodes receives the intervention.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=2, s_per_node=500, i_per_node=0, nticks=2, schedule=schedule)
    model.run()

    assert int(model.states.S[1, 0]) == 0
    assert int(model.states.V[1, 0]) == 500
    assert int(model.states.S[1, 1]) == 0
    assert int(model.states.V[1, 1]) == 500


# ---------------------------------------------------------------------------
# Recording: newly_vaccinated property
# ---------------------------------------------------------------------------


def test_newly_vaccinated_records_count_at_scheduled_tick() -> None:
    """Given a 1-node model with 1000 S and a Vaccination at tick 2 with coverage=1.0,
    when the model runs for 5 ticks, then model.nodes.newly_vaccinated[2, 0] == 1000.

    newly_vaccinated must capture the exact count of individuals vaccinated on
    the scheduled tick.  Failure means vaccination is not being recorded, which
    prevents downstream reporting and output analysis.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 2, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=0, nticks=5, schedule=schedule)
    model.run()

    assert int(model.nodes.newly_vaccinated[2, 0]) == 1000


def test_newly_vaccinated_zero_on_unscheduled_ticks() -> None:
    """Given a Vaccination scheduled only at tick 2, when the model runs for 5 ticks,
    then model.nodes.newly_vaccinated is 0 on all ticks other than tick 2.

    The intervention must not fire on unscheduled ticks.  Failure means the
    scheduler is treating the entry as an every-tick intervention.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 2, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=0, nticks=5, schedule=schedule)
    model.run()

    assert int(model.nodes.newly_vaccinated[0, 0]) == 0
    assert int(model.nodes.newly_vaccinated[1, 0]) == 0
    assert int(model.nodes.newly_vaccinated[3, 0]) == 0
    assert int(model.nodes.newly_vaccinated[4, 0]) == 0


# ---------------------------------------------------------------------------
# Population conservation
# ---------------------------------------------------------------------------


def test_population_conservation_after_vaccination() -> None:
    """Given a 1-node model with S=700, I=200, R=100 and a Vaccination at tick 1
    with coverage=0.6, when the model runs, then the total population (S+I+R+V)
    is identical at every tick.

    Vaccination must be a closed transfer: individuals leave one compartment and
    enter V; no individuals are created or destroyed.  Failure indicates a
    counting error in the binomial draw subtraction or V accumulation.
    """
    laser.core.random.seed(0)
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 1, "where": "*", "parameters": {"coverage": 0.6}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=700, i_per_node=200, nticks=4, schedule=schedule, r_per_node=100)
    model.run()

    total_0 = int(model.states[0].sum())
    for tick in range(1, 5):
        assert int(model.states[tick].sum()) == total_0, f"Population changed at tick {tick}"


# ---------------------------------------------------------------------------
# Persistence: vaccinated individuals accumulate in V across ticks
# ---------------------------------------------------------------------------


def test_vaccinated_individuals_persist_in_V_after_scheduled_tick() -> None:
    """Given a Vaccination at tick 0 moving all 1000 S to V, when the model runs
    for 3 additional ticks, then V stays at 1000 for ticks 1, 2, and 3.

    The carry-forward mechanism must preserve V across ticks; vaccinated
    individuals must not disappear silently.  Failure indicates V is reset or
    not carried forward correctly.
    """
    schedule = [{"who": ["S"], "what": "Vaccination", "when": 0, "where": "*", "parameters": {"coverage": 1.0}, "notes": ""}]
    model = _build_model(n_nodes=1, s_per_node=1000, i_per_node=0, nticks=4, schedule=schedule)
    model.run()

    assert int(model.states.V[1, 0]) == 1000
    assert int(model.states.V[2, 0]) == 1000
    assert int(model.states.V[3, 0]) == 1000
    assert int(model.states.V[4, 0]) == 1000


if __name__ == "__main__":
    pytest.main()
