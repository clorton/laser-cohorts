"""Tests for the Campaign scheduling component.

``Campaign`` is a model component that loads a list of intervention entries and
dispatches registered ``Intervention`` subclasses on the correct ticks, nodes,
and compartment states.

Each entry has six fields:
  - who:        ``"*"`` or list of state names
  - what:       registered intervention class name
  - when:       ``"*"`` (every tick), integer tick, or ``"YYYY-MM-DD"`` date
  - where:      ``"*"``, single integer node ID, or list of node IDs
  - parameters: arbitrary ``{key: value}`` dict
  - notes:      free-text string

Tests use ``RecordingIntervention``, a concrete ``Intervention`` subclass that
appends its call arguments to a shared list rather than modifying model state.
This decouples scheduling logic from epidemic dynamics and makes assertions
simple and deterministic.
"""

import json
import pytest
from pathlib import Path

from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Campaign, Intervention, Model
import laser.cohorts.SIR as SIR
from laser.generic.utils import ValuesMap


# ---------------------------------------------------------------------------
# Shared recording infrastructure
# ---------------------------------------------------------------------------

_calls: list[dict] = []


class RecordingIntervention(Intervention):
    """Intervention that appends its arguments to the module-level _calls list."""

    def execute(self, tick, who, where, params, notes):
        _calls.append({"tick": tick, "who": who, "where": where, "params": params, "notes": notes})


Campaign.register(RecordingIntervention)


@pytest.fixture(autouse=True)
def clear_calls():
    """Reset the shared call log before every test."""
    _calls.clear()
    yield


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Source loading: dict, list, JSON file, CSV file
# ---------------------------------------------------------------------------

def test_single_dict_source_fires_on_specified_tick() -> None:
    """Given a Campaign loaded from a single dict with when=2, when the model
    runs for 5 ticks, then the intervention fires exactly once on tick 2.

    Failure means the dict source is not parsed as a single-entry schedule, or
    the tick-specific dispatch is not working.
    """
    entry = {"who": "*", "what": "RecordingIntervention", "when": 2, "where": "*", "parameters": {}, "notes": ""}
    model = _make_model_with_schedule(entry, nticks=5)
    model.run()

    assert len(_calls) == 1
    assert _calls[0]["tick"] == 2


def test_list_source_fires_each_entry() -> None:
    """Given a Campaign loaded from a list of two entries at ticks 1 and 3, when
    the model runs for 5 ticks, then the intervention fires once on tick 1 and
    once on tick 3.

    Failure means list sources are not iterated or specific-tick dispatch is wrong.
    """
    schedule = [
        {"who": "*", "what": "RecordingIntervention", "when": 1, "where": "*", "parameters": {}, "notes": "first"},
        {"who": "*", "what": "RecordingIntervention", "when": 3, "where": "*", "parameters": {}, "notes": "second"},
    ]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 1
    assert _calls[0]["notes"] == "first"
    assert _calls[1]["tick"] == 3
    assert _calls[1]["notes"] == "second"


def _make_model_with_schedule(schedule, nticks: int = 5) -> Model:
    """Build a two-node SIR model with the given schedule entries."""
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    campaign = Campaign(model, schedule)
    betas = ValuesMap.from_scalar(0.0, nticks, n)
    r_recoveries = ValuesMap.from_scalar(0.0, nticks, n)
    model.components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
        campaign,
    ]
    return model


def test_json_file_source_loads_correctly(tmp_path: Path) -> None:
    """Given a Campaign loaded from a JSON file with two entries (ticks 0 and 4),
    when the model runs for 5 ticks, then both interventions fire on their
    respective ticks.

    Failure means JSON file loading or path resolution is broken.
    """
    schedule = [
        {"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": "json-a"},
        {"who": "*", "what": "RecordingIntervention", "when": 4, "where": "*", "parameters": {}, "notes": "json-b"},
    ]
    json_path = tmp_path / "schedule.json"
    json_path.write_text(json.dumps(schedule))

    model = _make_model_with_schedule(json_path)
    model.run()

    assert len(_calls) == 2
    ticks_fired = {c["tick"] for c in _calls}
    assert ticks_fired == {0, 4}


def test_csv_file_source_loads_correctly(tmp_path: Path) -> None:
    """Given a Campaign loaded from a CSV file with entries at ticks 1 and 2,
    when the model runs for 5 ticks, then both interventions fire on their
    respective ticks with the correct parameters and notes.

    Failure means CSV parsing, JSON-encoded parameter deserialization, or
    integer tick parsing from string is broken.
    """
    csv_content = (
        "who,what,when,where,parameters,notes\n"
        '*,RecordingIntervention,1,*,"{""value"": 7}",csv-first\n'
        '*,RecordingIntervention,2,*,"{}",csv-second\n'
    )
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(csv_content)

    model = _make_model_with_schedule(csv_path)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 1
    assert _calls[0]["params"] == {"value": 7}
    assert _calls[0]["notes"] == "csv-first"
    assert _calls[1]["tick"] == 2
    assert _calls[1]["notes"] == "csv-second"


# ---------------------------------------------------------------------------
# when field variants
# ---------------------------------------------------------------------------

def test_when_star_fires_every_tick() -> None:
    """Given a single entry with when="*", when the model runs for 4 ticks, then
    the intervention fires on every tick (4 times).

    Failure means the every-tick branch in the scheduler is not triggering or is
    being deduped incorrectly.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": "*", "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=4)
    model.run()

    assert len(_calls) == 4
    assert [c["tick"] for c in _calls] == [0, 1, 2, 3]


def test_when_integer_fires_only_on_that_tick() -> None:
    """Given a single entry with when=3, when the model runs for 6 ticks, then
    the intervention fires exactly once, on tick 3.

    Failure means integer-tick dispatch fires on multiple ticks or the wrong tick.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 3, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=6)
    model.run()

    assert len(_calls) == 1
    assert _calls[0]["tick"] == 3


def test_when_date_fires_on_correct_tick() -> None:
    """Given a schedule entry with when="2020-02-01" and start_date="2020-01-01",
    when the model runs for 40 ticks, then the intervention fires exactly once
    on tick 31 (31 days from the start).

    Failure means date-to-tick conversion is off by one or ignoring start_date.
    """
    nticks = 40
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    schedule = [
        {"who": "*", "what": "RecordingIntervention", "when": "2020-02-01",
         "where": "*", "parameters": {}, "notes": ""},
    ]
    campaign = Campaign(model, schedule, start_date="2020-01-01")
    betas = ValuesMap.from_scalar(0.0, nticks, n)
    r_recoveries = ValuesMap.from_scalar(0.0, nticks, n)
    model.components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
        campaign,
    ]
    model.run()

    assert len(_calls) == 1
    assert _calls[0]["tick"] == 31  # 31 days: Jan has 31 days, so Feb 1 = day 31


def test_when_no_entries_match_fires_nothing() -> None:
    """Given a single entry with when=10, when the model runs for 5 ticks (so
    tick 10 is never reached), then the intervention never fires.

    Failure means the scheduler fires at tick 10 even when it is past nticks,
    or fires spuriously on other ticks.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 10, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    assert len(_calls) == 0


# ---------------------------------------------------------------------------
# where field variants
# ---------------------------------------------------------------------------

def test_where_star_passes_none_to_intervention() -> None:
    """Given a schedule entry with where="*", when the intervention fires, then
    the ``where`` argument received by the intervention is None (meaning all nodes).

    Failure means "*" is being passed through as a string rather than normalised
    to None.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["where"] is None


def test_where_single_int_passes_list_of_one() -> None:
    """Given a schedule entry with where=1, when the intervention fires, then
    the ``where`` argument is [1].

    Failure means a scalar node ID is not wrapped in a list.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": 1, "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["where"] == [1]


def test_where_list_passes_through_unchanged() -> None:
    """Given a schedule entry with where=[0, 1], when the intervention fires, then
    the ``where`` argument is [0, 1].

    Failure means the list is being re-wrapped or truncated.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": [0, 1], "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["where"] == [0, 1]


# ---------------------------------------------------------------------------
# who field variants
# ---------------------------------------------------------------------------

def test_who_star_passes_none_to_intervention() -> None:
    """Given a schedule entry with who="*", when the intervention fires, then
    the ``who`` argument received by the intervention is None (meaning all states).

    Failure means "*" is passed through as a string rather than normalised to None.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["who"] is None


def test_who_list_passes_through_to_intervention() -> None:
    """Given a schedule entry with who=["S", "R"], when the intervention fires,
    then the ``who`` argument is ["S", "R"].

    Failure means the list is dropped or replaced with None.
    """
    schedule = [{"who": ["S", "R"], "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["who"] == ["S", "R"]


# ---------------------------------------------------------------------------
# parameters and notes forwarding
# ---------------------------------------------------------------------------

def test_parameters_forwarded_to_intervention() -> None:
    """Given a schedule entry with parameters={"coverage": 0.9, "round": 2}, when
    the intervention fires, then params received exactly matches that dict.

    Failure means parameters are dropped, shallow-copied incorrectly, or merged
    with other data.
    """
    params = {"coverage": 0.9, "round": 2}
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": params, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["params"] == params


def test_notes_forwarded_to_intervention() -> None:
    """Given a schedule entry with notes="Round 1 campaign", when the
    intervention fires, then the notes string received is exactly
    "Round 1 campaign".

    Failure means notes are dropped or replaced with an empty string.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": "Round 1 campaign"}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["notes"] == "Round 1 campaign"


# ---------------------------------------------------------------------------
# Multiple interventions on the same tick
# ---------------------------------------------------------------------------

def test_multiple_entries_on_same_tick_all_fire() -> None:
    """Given two schedule entries both with when=2, when the model runs, then
    both interventions fire on tick 2 (in schedule order).

    Failure means only one intervention fires when two are scheduled at the same
    tick, indicating entries are being deduplicated or the list is truncated.
    """
    schedule = [
        {"who": "*", "what": "RecordingIntervention", "when": 2, "where": "*", "parameters": {}, "notes": "alpha"},
        {"who": "*", "what": "RecordingIntervention", "when": 2, "where": "*", "parameters": {}, "notes": "beta"},
    ]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    tick2_calls = [c for c in _calls if c["tick"] == 2]
    assert len(tick2_calls) == 2
    assert tick2_calls[0]["notes"] == "alpha"
    assert tick2_calls[1]["notes"] == "beta"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_unknown_intervention_name_raises_key_error() -> None:
    """Given a schedule entry with what="DoesNotExist" (not registered), when
    the model runs and that tick is reached, then a KeyError is raised.

    Failure means an unregistered intervention silently does nothing, which
    would cause misunderstood simulation results.
    """
    schedule = [{"who": "*", "what": "DoesNotExist", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=3)
    with pytest.raises(KeyError, match="DoesNotExist"):
        model.run()


def test_mixed_date_and_int_when_raises_value_error() -> None:
    """Given a schedule with one date-based when ("2020-01-10") and one integer
    when (5), when the Campaign is constructed, then a ValueError is raised.

    Mixing date strings and integer ticks in the same schedule is ambiguous and
    must be caught early.  Failure means the mixed schedule runs without error,
    likely producing wrong tick assignments.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 10, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    schedule = [
        {"who": "*", "what": "RecordingIntervention", "when": "2020-01-10", "where": "*", "parameters": {}, "notes": ""},
        {"who": "*", "what": "RecordingIntervention", "when": 5, "where": "*", "parameters": {}, "notes": ""},
    ]
    with pytest.raises(ValueError, match="mix"):
        Campaign(model, schedule, start_date="2020-01-01")


def test_date_when_without_start_date_raises_value_error() -> None:
    """Given a schedule with a date-based when but no start_date argument,
    when the Campaign is constructed, then a ValueError is raised.

    Without start_date there is no reference point to convert dates to ticks.
    Failure means the Campaign silently miscomputes tick offsets.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 10, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    schedule = [{"who": "*", "what": "RecordingIntervention", "when": "2020-01-10", "where": "*", "parameters": {}, "notes": ""}]
    with pytest.raises(ValueError, match="start_date"):
        Campaign(model, schedule)


def test_unsupported_file_format_raises_value_error() -> None:
    """Given a Campaign constructed with a path ending in .txt (unsupported),
    when the Campaign is constructed, then a ValueError is raised.

    Failure means an unsupported file type is silently ignored or raises an
    unintelligible error.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    with pytest.raises(ValueError, match="Unsupported source"):
        Campaign(model, Path("/tmp/schedule.txt"))


# ---------------------------------------------------------------------------
# CSV complex fields: who and where as JSON in CSV
# ---------------------------------------------------------------------------

def test_csv_where_as_json_list(tmp_path: Path) -> None:
    """Given a CSV schedule with where="[0]" (a JSON list in the CSV field),
    when the intervention fires, then where received is [0].

    Failure means the JSON list in the CSV where column is left as a raw string
    rather than parsed.
    """
    csv_content = (
        "who,what,when,where,parameters,notes\n"
        '*,RecordingIntervention,0,"[0]","{}",\n'
    )
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(csv_content)

    model = _make_model_with_schedule(csv_path, nticks=3)
    model.run()

    assert _calls[0]["where"] == [0]


def test_csv_who_as_json_list(tmp_path: Path) -> None:
    """Given a CSV schedule with who='["S"]' (a JSON list in the CSV field),
    when the intervention fires, then who received is ["S"].

    Failure means the JSON list in the CSV who column is left as a raw string.
    """
    csv_content = (
        "who,what,when,where,parameters,notes\n"
        '"""[""S""]"",RecordingIntervention,0,*,"{}",\n'
    )
    # Use an alternate approach: write via json.dumps to avoid escaping issues
    import csv as csv_mod
    csv_path = tmp_path / "schedule.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["who", "what", "when", "where", "parameters", "notes"])
        writer.writeheader()
        writer.writerow({
            "who": json.dumps(["S"]),
            "what": "RecordingIntervention",
            "when": 0,
            "where": "*",
            "parameters": "{}",
            "notes": "",
        })

    model = _make_model_with_schedule(csv_path, nticks=3)
    model.run()

    assert _calls[0]["who"] == ["S"]
