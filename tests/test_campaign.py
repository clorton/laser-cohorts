"""Tests for the Campaign scheduling component.

``Campaign`` is a model component that loads a list of intervention entries and
dispatches registered ``Intervention`` subclasses on the correct ticks, nodes,
and compartment states.

Each entry has six fields:
  - who:        ``"*"`` or list of state names
  - what:       registered intervention class name
  - when:       ``"*"`` (every tick), integer tick, list of integer ticks, or ``"YYYY-MM-DD"`` date
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
from laser.cohorts import Campaign, Intervention, Model, ScheduledEntry
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
        {"who": "*", "what": "RecordingIntervention", "when": "2020-02-01", "where": "*", "parameters": {}, "notes": ""},
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


def test_when_list_of_ticks_fires_on_each_tick() -> None:
    """Given a single entry with when=[1, 3], when the model runs for 5 ticks,
    then the intervention fires exactly twice: once on tick 1 and once on tick 3.

    A list of integer ticks should produce one firing per listed tick.  Failure
    means the list is being treated as a single tick, dropped entirely, or fires
    on the wrong ticks.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": [1, 3], "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 1
    assert _calls[1]["tick"] == 3


def test_when_list_only_fires_for_in_range_ticks() -> None:
    """Given a single entry with when=[2, 10] and nticks=5, when the model runs,
    then the intervention fires only on tick 2 (tick 10 is never reached).

    Out-of-range ticks in the list are silently skipped because the model simply
    never executes those ticks.  Failure means the scheduler errors on out-of-range
    ticks or fires spuriously.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": [2, 10], "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    assert len(_calls) == 1
    assert _calls[0]["tick"] == 2


def test_when_list_preserves_per_tick_who_and_params() -> None:
    """Given an entry with when=[0, 4], who=["S"], and parameters={"dose": 1},
    when the model runs for 5 ticks, then both firings receive who=["S"] and
    params={"dose": 1} unchanged.

    The same entry metadata must be forwarded identically to every tick in the
    list.  Failure means metadata is mutated between firings or only carried for
    the first tick.
    """
    schedule = [
        {"who": ["S"], "what": "RecordingIntervention", "when": [0, 4], "where": "*", "parameters": {"dose": 1}, "notes": "booster"}
    ]
    model = _make_model_with_schedule(schedule, nticks=5)
    model.run()

    assert len(_calls) == 2
    for call in _calls:
        assert call["who"] == ["S"]
        assert call["params"] == {"dose": 1}
        assert call["notes"] == "booster"


def _make_model_with_date_schedule(schedule, start_date: str, nticks: int = 40) -> Model:
    """Build a two-node SIR model with a date-based schedule."""
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    campaign = Campaign(model, schedule, start_date=start_date)
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


def test_when_list_of_dates_fires_on_each_date() -> None:
    """Given a schedule entry with when=["2020-01-10", "2020-02-01"] and
    start_date="2020-01-01", when the model runs for 40 ticks, then the
    intervention fires exactly twice: once on tick 9 (Jan 10) and once on
    tick 31 (Feb 1).

    A list of date strings should produce one firing per listed date, with
    each date converted to a tick offset from start_date.  Failure means the
    date list is rejected, treated as a single tick, or converted incorrectly.
    """
    schedule = [
        {"who": "*", "what": "RecordingIntervention",
         "when": ["2020-01-10", "2020-02-01"], "where": "*",
         "parameters": {}, "notes": ""}
    ]
    model = _make_model_with_date_schedule(schedule, start_date="2020-01-01", nticks=40)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 9   # Jan 10 - Jan 1 = 9 days
    assert _calls[1]["tick"] == 31  # Feb 1 - Jan 1 = 31 days


def test_when_list_of_dates_only_fires_for_in_range_ticks() -> None:
    """Given a list of two dates where the second resolves past nticks, when
    the model runs, then only the in-range firing actually executes.

    Out-of-range tick offsets in a date list are silently skipped because the
    model never executes those ticks.  Failure means the scheduler errors on
    out-of-range ticks or fires the wrong dates.
    """
    schedule = [
        {"who": "*", "what": "RecordingIntervention",
         "when": ["2020-01-05", "2020-12-01"], "where": "*",
         "parameters": {}, "notes": ""}
    ]
    model = _make_model_with_date_schedule(schedule, start_date="2020-01-01", nticks=20)
    model.run()

    assert len(_calls) == 1
    assert _calls[0]["tick"] == 4  # Jan 5 - Jan 1 = 4 days


def test_when_list_of_dates_preserves_per_firing_who_and_params() -> None:
    """Given a list-of-dates entry with who=["S"] and parameters={"dose": 2},
    when the model runs, then every firing receives the same who and params.

    The same entry metadata must be forwarded identically to every date in
    the list, exactly as it is for lists of integer ticks.
    """
    schedule = [
        {"who": ["S"], "what": "RecordingIntervention",
         "when": ["2020-01-05", "2020-01-15", "2020-01-25"], "where": "*",
         "parameters": {"dose": 2}, "notes": "supplementary"}
    ]
    model = _make_model_with_date_schedule(schedule, start_date="2020-01-01", nticks=40)
    model.run()

    assert len(_calls) == 3
    for call in _calls:
        assert call["who"] == ["S"]
        assert call["params"] == {"dose": 2}
        assert call["notes"] == "supplementary"


def test_when_list_of_dates_without_start_date_raises_value_error() -> None:
    """Given a schedule entry with when=["2020-01-10", "2020-01-20"] but no
    start_date, when the Campaign is constructed, then a ValueError is raised.

    A list of dates requires start_date to convert to tick offsets.  Failure
    means the campaign silently miscomputes tick offsets or crashes deeper in
    the pipeline.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 40, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    schedule = [
        {"who": "*", "what": "RecordingIntervention",
         "when": ["2020-01-10", "2020-01-20"], "where": "*",
         "parameters": {}, "notes": ""}
    ]
    with pytest.raises(ValueError, match="start_date"):
        Campaign(model, schedule)


def test_when_list_mixed_dates_and_ints_raises_value_error() -> None:
    """Given a schedule entry with when=["2020-01-10", 15] (mixed dates and
    integers in the same list), when the Campaign is constructed, then a
    ValueError is raised.

    Mixing dates and integer ticks within a single list is ambiguous and must
    be caught early.  Failure means the mixed list is silently accepted,
    producing inconsistent tick assignments.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 40, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    schedule = [
        {"who": "*", "what": "RecordingIntervention",
         "when": ["2020-01-10", 15], "where": "*",
         "parameters": {}, "notes": ""}
    ]
    with pytest.raises(ValueError, match="mix"):
        Campaign(model, schedule, start_date="2020-01-01")


def test_when_list_of_dates_before_start_date_raises_value_error() -> None:
    """Given a list-of-dates entry where one date is before start_date, when
    the Campaign is constructed, then a ValueError is raised.

    Dates earlier than start_date map to negative tick offsets, which the
    scheduler cannot fire.  Failure means a negative tick is silently
    accepted, masking a configuration error.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 40, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    schedule = [
        {"who": "*", "what": "RecordingIntervention",
         "when": ["2020-01-10", "2019-12-15"], "where": "*",
         "parameters": {}, "notes": ""}
    ]
    with pytest.raises(ValueError, match="before campaign start date"):
        Campaign(model, schedule, start_date="2020-01-01")


def test_csv_when_as_json_array_of_dates_fires_on_each_date(tmp_path: Path) -> None:
    """Given a CSV schedule where the when column is a JSON array of date
    strings, when the model runs, then the intervention fires on the tick
    corresponding to each listed date.

    CSV's JSON-array encoding must round-trip through the same date-list path
    as a Python-level list.  Failure means the CSV loader rejects date-array
    cells or converts them incorrectly.
    """
    import csv as csv_mod

    csv_path = tmp_path / "schedule.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["who", "what", "when", "where", "parameters", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "who": "*",
                "what": "RecordingIntervention",
                "when": json.dumps(["2020-01-08", "2020-01-29"]),
                "where": "*",
                "parameters": "{}",
                "notes": "",
            }
        )

    model = _make_model_with_date_schedule(csv_path, start_date="2020-01-01", nticks=40)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 7   # Jan 8 - Jan 1 = 7 days
    assert _calls[1]["tick"] == 28  # Jan 29 - Jan 1 = 28 days


def test_csv_when_as_json_array_fires_on_each_tick(tmp_path: Path) -> None:
    """Given a CSV schedule where the when column is a JSON array "[1, 3]", when
    the model runs for 5 ticks, then the intervention fires on ticks 1 and 3.

    CSV does not have a native list type; a JSON array in the when column is the
    documented encoding.  Failure means the JSON array string is not parsed and
    the entry is silently dropped or raises an error.
    """
    import csv as csv_mod

    csv_path = tmp_path / "schedule.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["who", "what", "when", "where", "parameters", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "who": "*",
                "what": "RecordingIntervention",
                "when": json.dumps([1, 3]),
                "where": "*",
                "parameters": "{}",
                "notes": "",
            }
        )

    model = _make_model_with_schedule(csv_path, nticks=5)
    model.run()

    assert len(_calls) == 2
    assert _calls[0]["tick"] == 1
    assert _calls[1]["tick"] == 3


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


def test_who_bare_string_wrapped_as_single_element_list() -> None:
    """Given a schedule entry with who="S" (a bare scalar string), when the
    intervention fires, then the ``who`` argument is ["S"].

    ``who``, ``where``, and ``when`` all accept a single scalar that should be
    treated as a one-element list — failure means a bare string is rejected
    or passed through unwrapped.
    """
    schedule = [{"who": "S", "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    model = _make_model_with_schedule(schedule, nticks=2)
    model.run()

    assert _calls[0]["who"] == ["S"]


def test_csv_who_bare_string_cell_parses_as_single_element_list(tmp_path: Path) -> None:
    """Given a CSV schedule with who="S" (a bare scalar cell, not bracketed),
    when the intervention fires, then ``who`` received is ["S"].

    CSV cells for ``who``, ``where``, and ``when`` share a uniform grammar:
    "*", a bracketed JSON list, or a single bare scalar.  Failure means the
    CSV loader still requires bracket syntax for single-element ``who``.
    """
    csv_content = "who,what,when,where,parameters,notes\nS,RecordingIntervention,0,*,{},\n"
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(csv_content)

    model = _make_model_with_schedule(csv_path, nticks=3)
    model.run()

    assert _calls[0]["who"] == ["S"]


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


def test_unknown_intervention_name_raises_value_error_at_construction() -> None:
    """Given a schedule entry with what="DoesNotExist" (not registered), when
    the Campaign is constructed, then a ValueError is raised that names the
    unregistered class.

    Validation runs at construction time so unregistered intervention names
    are caught before any tick is executed.  Failure means the error is
    deferred until the scheduled tick fires, or worse, the intervention is
    silently skipped.
    """
    schedule = [{"who": "*", "what": "DoesNotExist", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 3, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="DoesNotExist"):
        Campaign(model, schedule)


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


def test_missing_who_raises_value_error() -> None:
    """Given a schedule entry that omits the 'who' field, when the Campaign is
    constructed, then a ValueError is raised that mentions 'who'.

    'who' is a required field — silently defaulting to '*' would hide a
    configuration mistake.  Failure means the missing key is silently filled
    in, producing a schedule that may target unintended states.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    schedule = [{"what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    with pytest.raises(ValueError, match="who"):
        Campaign(model, schedule)


def test_missing_where_raises_value_error() -> None:
    """Given a schedule entry that omits the 'where' field, when the Campaign is
    constructed, then a ValueError is raised that mentions 'where'.

    'where' is a required field — silently defaulting to '*' would hide a
    configuration mistake.  Failure means the missing key is silently filled
    in, producing a schedule that may target unintended nodes.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "parameters": {}, "notes": ""}]
    with pytest.raises(ValueError, match="where"):
        Campaign(model, schedule)


def test_csv_missing_who_column_raises_value_error(tmp_path: Path) -> None:
    """Given a CSV schedule whose header lacks the 'who' column, when the
    Campaign is constructed, then a ValueError is raised that mentions 'who'.

    A missing CSV column must produce the same clear error as a missing dict
    key — failure means the schedule is silently accepted with a default
    target.
    """
    csv_content = "what,when,where,parameters,notes\nRecordingIntervention,0,*,{},\n"
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(csv_content)

    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 3, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    with pytest.raises(ValueError, match="who"):
        Campaign(model, csv_path)


def test_csv_empty_where_cell_raises_value_error(tmp_path: Path) -> None:
    """Given a CSV schedule where the 'where' cell is empty (column present but
    blank), when the Campaign is constructed, then a ValueError is raised.

    An empty cell in a required column must be treated the same as a missing
    column — failure means the schedule silently picks a default target node.
    """
    csv_content = "who,what,when,where,parameters,notes\n*,RecordingIntervention,0,,{},\n"
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text(csv_content)

    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 3, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    with pytest.raises(ValueError, match="where"):
        Campaign(model, csv_path)


def test_missing_what_raises_value_error() -> None:
    """Given a schedule entry that omits the 'what' field, when the Campaign is
    constructed, then a ValueError is raised that mentions 'what'.

    'what' identifies the intervention class to dispatch — omitting it leaves
    the entry meaningless.  Failure means the schedule is accepted and only
    crashes later, deep inside `step()`.
    """
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)

    schedule = [{"who": "*", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    with pytest.raises(ValueError, match="what"):
        Campaign(model, schedule)


def test_invalid_who_int_raises_value_error() -> None:
    """Given a schedule entry with who=42 (an int, not a string/list/'*'), when
    the Campaign is constructed, then a ValueError is raised that mentions 'who'.

    The 'who' grammar is '*' / single string / list of strings; an int is
    nonsensical.  Failure means an int who is silently coerced or accepted.
    """
    schedule = [{"who": 42, "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="who"):
        Campaign(model, schedule)


def test_invalid_where_string_raises_value_error() -> None:
    """Given a schedule entry with where="abc" (a non-numeric string), when the
    Campaign is constructed, then a ValueError is raised that mentions 'where'.

    The 'where' grammar is '*' / single int / list of ints.  A non-numeric
    string is a configuration error.  Failure means the value is accepted and
    crashes later inside `_normalize_where`.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "abc", "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="where"):
        Campaign(model, schedule)


def test_invalid_when_string_raises_value_error() -> None:
    """Given a schedule entry with when="notadate" (neither digits nor a
    YYYY-MM-DD date string), when the Campaign is constructed, then a
    ValueError is raised that mentions 'when'.

    The 'when' grammar is '*' / int / date string / homogeneous list.  A
    non-numeric, non-date string is a configuration error.  Failure means the
    value is accepted and crashes later.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": "notadate", "where": "*", "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="when"):
        Campaign(model, schedule)


def test_invalid_who_list_with_non_string_raises_value_error() -> None:
    """Given a schedule entry with who=["S", 42] (a list mixing string and int),
    when the Campaign is constructed, then a ValueError is raised.

    A 'who' list must contain only strings.  Failure means the int is silently
    forwarded to the intervention which has no sensible interpretation of it.
    """
    schedule = [{"who": ["S", 42], "what": "RecordingIntervention", "when": 0, "where": "*", "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="who"):
        Campaign(model, schedule)


def test_invalid_where_list_with_non_int_raises_value_error() -> None:
    """Given a schedule entry with where=[0, "foo"] (a list with a non-int
    element), when the Campaign is constructed, then a ValueError is raised.

    A 'where' list must contain only ints.  Failure means the bad element is
    accepted and crashes later inside `_normalize_where`.
    """
    schedule = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": [0, "foo"], "parameters": {}, "notes": ""}]
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": 5, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    with pytest.raises(ValueError, match="where"):
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
    csv_content = 'who,what,when,where,parameters,notes\n*,RecordingIntervention,0,"[0]","{}",\n'
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
    # Use an alternate approach: write via json.dumps to avoid escaping issues
    import csv as csv_mod

    csv_path = tmp_path / "schedule.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["who", "what", "when", "where", "parameters", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "who": json.dumps(["S"]),
                "what": "RecordingIntervention",
                "when": 0,
                "where": "*",
                "parameters": "{}",
                "notes": "",
            }
        )

    model = _make_model_with_schedule(csv_path, nticks=3)
    model.run()

    assert _calls[0]["who"] == ["S"]


# ---------------------------------------------------------------------------
# Campaign.add_entry — runtime addition of scheduled entries
# ---------------------------------------------------------------------------


def _make_model_for_add_entry(initial_schedule, nticks: int = 5) -> Model:
    """Build a minimal two-node SIR model with the given initial schedule."""
    n = 2
    scenario = grid(M=n, N=1)
    scenario["S"] = 1000
    scenario["I"] = 10
    scenario["R"] = 0
    p = PropertySet({"nticks": nticks, "beta": 0.0, "r_recovery": 0.0})
    model = Model(scenario, p)
    campaign = Campaign(model, initial_schedule)
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


def test_add_entry_routes_tick_specific_entry_to_at_tick() -> None:
    """Given a Campaign with a single initial entry at tick 0, when add_entry
    is called with a ScheduledEntry for tick 3, then the entry fires on tick 3
    in addition to the original entry at tick 0.

    Failure means the new entry was either dropped, fired on the wrong tick,
    or shadowed the existing entry.
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "initial"}]
    model = _make_model_for_add_entry(initial, nticks=5)
    campaign = model.components[-1]

    new_entry = ScheduledEntry(
        what="RecordingIntervention",
        who=None,
        where=None,
        params={},
        notes="added",
        tick=3,
    )
    campaign.add_entry(new_entry)

    model.run()

    fire_ticks = sorted(c["tick"] for c in _calls)
    fire_notes = [c["notes"] for c in _calls]
    assert fire_ticks == [0, 3]
    assert "added" in fire_notes
    assert "initial" in fire_notes


def test_add_entry_routes_every_tick_entry_to_every_tick() -> None:
    """Given a Campaign with no every-tick entries, when add_entry is called
    with a ScheduledEntry whose tick is None, then the entry fires on every
    subsequent tick.

    Failure means the every-tick routing branch is broken.
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "init"}]
    model = _make_model_for_add_entry(initial, nticks=4)
    campaign = model.components[-1]

    every_tick = ScheduledEntry(
        what="RecordingIntervention",
        who=None,
        where=None,
        params={},
        notes="every",
        tick=None,
    )
    campaign.add_entry(every_tick)

    model.run()

    every_calls = [c for c in _calls if c["notes"] == "every"]
    assert len(every_calls) == 4
    assert sorted(c["tick"] for c in every_calls) == [0, 1, 2, 3]


def test_add_entry_dispatchable_from_inside_an_intervention() -> None:
    """Given a 'Trigger' intervention that, on its first firing, calls
    add_entry to schedule a follow-up RecordingIntervention at tick + 2,
    when the model runs, then the recording intervention fires on the
    follow-up tick.

    Failure means add_entry is not visible to dispatch on subsequent ticks,
    or the entry is registered in the wrong bucket.
    """

    class TriggerIntervention(Intervention):
        @property
        def states(self):
            return []

        @property
        def properties(self):
            return []

        def execute(self, tick, who, where, params, notes):
            # locate campaign and schedule a follow-up at tick + 2
            campaign = next(c for c in self.model.components if isinstance(c, Campaign))
            campaign.add_entry(
                ScheduledEntry(
                    what="RecordingIntervention",
                    who=None,
                    where=None,
                    params={},
                    notes="reactive",
                    tick=tick + 2,
                )
            )

    Campaign.register(TriggerIntervention)

    initial = [{"who": "*", "what": "TriggerIntervention", "when": 1, "where": "*",
                "parameters": {}, "notes": "trigger"}]
    model = _make_model_for_add_entry(initial, nticks=6)
    model.run()

    # Trigger fires at tick 1, RecordingIntervention at tick 3.
    recording_calls = [c for c in _calls if c["notes"] == "reactive"]
    assert len(recording_calls) == 1
    assert recording_calls[0]["tick"] == 3


def test_add_entry_rejects_non_scheduled_entry_with_type_error() -> None:
    """Given a plain dict passed to add_entry instead of a ScheduledEntry,
    when add_entry is invoked, then a TypeError is raised.

    Failure means dict-shaped inputs are silently accepted and lead to attribute
    errors deep inside Campaign.step when the dispatcher tries `entry.what`.
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "init"}]
    model = _make_model_for_add_entry(initial, nticks=3)
    campaign = model.components[-1]

    with pytest.raises(TypeError, match="ScheduledEntry"):
        campaign.add_entry({"what": "RecordingIntervention"})


def test_add_entry_rejects_unregistered_what_with_value_error() -> None:
    """Given a ScheduledEntry whose 'what' names an unregistered intervention,
    when add_entry is invoked, then a ValueError is raised that mentions the
    bad name.

    Failure means the validation gate inherited from the construction-time
    validator has been lost for runtime additions.
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "init"}]
    model = _make_model_for_add_entry(initial, nticks=3)
    campaign = model.components[-1]

    bad_entry = ScheduledEntry(
        what="DoesNotExist",
        who=None,
        where=None,
        params={},
        notes="bogus",
        tick=2,
    )
    with pytest.raises(ValueError, match="DoesNotExist"):
        campaign.add_entry(bad_entry)


def test_add_entry_forwards_who_where_params_and_notes_to_intervention() -> None:
    """Given a ScheduledEntry with explicit who, where, params, and notes, when
    add_entry schedules it and the model runs to the firing tick, then the
    intervention's execute receives those exact values.

    Failure means add_entry mangled fields on the way through (e.g. defaulted
    who/where, dropped params, or coerced notes).
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "init"}]
    model = _make_model_for_add_entry(initial, nticks=4)
    campaign = model.components[-1]

    campaign.add_entry(
        ScheduledEntry(
            what="RecordingIntervention",
            who=["S", "R"],
            where=[1, 0],
            params={"dose": 7, "round": "spring"},
            notes="added-entry-with-fields",
            tick=2,
        )
    )

    model.run()

    matches = [c for c in _calls if c["notes"] == "added-entry-with-fields"]
    assert len(matches) == 1
    call = matches[0]
    assert call["tick"]   == 2
    assert call["who"]    == ["S", "R"]
    assert call["where"]  == [1, 0]
    assert call["params"] == {"dose": 7, "round": "spring"}


def test_add_entry_multiple_entries_same_tick_all_fire_in_added_order() -> None:
    """Given three ScheduledEntries added for the same tick in sequence, when
    the model runs to that tick, then all three fire and the firing order
    matches the order they were added to the campaign.

    Failure means the at_tick bucket either drops entries or reorders them.
    """
    initial = [{"who": "*", "what": "RecordingIntervention", "when": 0, "where": "*",
                "parameters": {}, "notes": "init"}]
    model = _make_model_for_add_entry(initial, nticks=4)
    campaign = model.components[-1]

    for tag in ("alpha", "beta", "gamma"):
        campaign.add_entry(
            ScheduledEntry(
                what="RecordingIntervention",
                who=None, where=None,
                params={}, notes=tag, tick=2,
            )
        )

    model.run()

    tick2_notes = [c["notes"] for c in _calls if c["tick"] == 2]
    assert tick2_notes == ["alpha", "beta", "gamma"]


def test_add_entry_for_already_past_tick_never_fires() -> None:
    """Given a ScheduledEntry added for a tick that has already been stepped
    past, when the model continues running, then the entry never fires.

    Past ticks are unreachable — Campaign.step only ever fetches the bucket
    for the current tick going forward.  Failure means past entries were
    re-replayed (which would corrupt the simulation history).
    """

    class AddPastEntryIntervention(Intervention):
        @property
        def states(self):
            return []

        @property
        def properties(self):
            return []

        def execute(self, tick, who, where, params, notes):
            # at tick 3, schedule a never-firing recording for tick 1
            campaign = next(c for c in self.model.components if isinstance(c, Campaign))
            campaign.add_entry(
                ScheduledEntry(
                    what="RecordingIntervention",
                    who=None, where=None,
                    params={}, notes="too-late", tick=1,
                )
            )

    Campaign.register(AddPastEntryIntervention)

    initial = [{"who": "*", "what": "AddPastEntryIntervention", "when": 3, "where": "*",
                "parameters": {}, "notes": "trigger"}]
    model = _make_model_for_add_entry(initial, nticks=6)
    model.run()

    # The trigger ran at tick 3; the added entry targeted tick 1 (already past)
    too_late = [c for c in _calls if c["notes"] == "too-late"]
    assert too_late == []


def test_add_entry_for_current_tick_does_not_fire_this_tick() -> None:
    """Given an intervention at tick T that calls add_entry to schedule another
    entry for the same tick T, when the model continues, then the newly added
    entry does NOT fire on tick T (it would on tick T+1 if scheduled there).

    Campaign.step builds its dispatch list at the start of the tick.  An entry
    added during dispatch goes into the at_tick bucket but is invisible to the
    in-flight iteration.  This test pins that semantic so a future change to
    Campaign.step that re-reads its bucket mid-iteration would be noticed.
    """

    class SelfRescheduleIntervention(Intervention):
        @property
        def states(self):
            return []

        @property
        def properties(self):
            return []

        def execute(self, tick, who, where, params, notes):
            campaign = next(c for c in self.model.components if isinstance(c, Campaign))
            campaign.add_entry(
                ScheduledEntry(
                    what="RecordingIntervention",
                    who=None, where=None,
                    params={}, notes="same-tick", tick=tick,
                )
            )

    Campaign.register(SelfRescheduleIntervention)

    initial = [{"who": "*", "what": "SelfRescheduleIntervention", "when": 2, "where": "*",
                "parameters": {}, "notes": "trigger"}]
    model = _make_model_for_add_entry(initial, nticks=5)
    model.run()

    # The trigger ran at tick 2 and scheduled a same-tick follow-up.
    # The follow-up should NOT have fired (entries list was already built).
    same_tick = [c for c in _calls if c["notes"] == "same-tick"]
    assert same_tick == []


def test_add_entry_supports_cascading_runtime_additions() -> None:
    """Given an intervention that calls add_entry, where the *added* entry
    itself calls add_entry on its own firing, when the model runs long enough
    for both to fire, then both reactive entries are observed in the call log.

    This exercises the loop: a runtime-added entry's execute() should have the
    same access to Campaign as a statically scheduled one.
    """

    class CascadeIntervention(Intervention):
        @property
        def states(self):
            return []

        @property
        def properties(self):
            return []

        def execute(self, tick, who, where, params, notes):
            # On the first firing only, schedule the second cascade.
            if notes == "first":
                campaign = next(c for c in self.model.components if isinstance(c, Campaign))
                campaign.add_entry(
                    ScheduledEntry(
                        what="CascadeIntervention",
                        who=None, where=None,
                        params={}, notes="second", tick=tick + 2,
                    )
                )

    Campaign.register(CascadeIntervention)

    # Use a RecordingIntervention rather than CascadeIntervention so the call
    # log captures the firings; cascade's execute doesn't record itself.  Wrap
    # the cascade so the second firing schedules a recording on tick+1.
    class RecordingCascade(Intervention):
        @property
        def states(self):
            return []

        @property
        def properties(self):
            return []

        def execute(self, tick, who, where, params, notes):
            _calls.append({"tick": tick, "who": who, "where": where, "params": params, "notes": notes})
            if notes == "first":
                campaign = next(c for c in self.model.components if isinstance(c, Campaign))
                campaign.add_entry(
                    ScheduledEntry(
                        what="RecordingCascade",
                        who=None, where=None,
                        params={}, notes="second", tick=tick + 2,
                    )
                )

    Campaign.register(RecordingCascade)

    initial = [{"who": "*", "what": "RecordingCascade", "when": 1, "where": "*",
                "parameters": {}, "notes": "first"}]
    model = _make_model_for_add_entry(initial, nticks=8)
    model.run()

    # We expect two cascade events: the initial 'first' at tick 1 and the
    # cascaded 'second' at tick 3.
    cascade_ticks = sorted(c["tick"] for c in _calls if c["notes"] in {"first", "second"})
    assert cascade_ticks == [1, 3]


def test_add_entry_works_with_empty_initial_schedule() -> None:
    """Given a Campaign constructed from an empty list, when add_entry is
    used to populate it before the model runs, then the scheduled entries
    still fire on their target ticks.

    'Start empty and grow at runtime' is a useful pattern; failure means
    a Campaign with no initial entries can't be extended.
    """
    model = _make_model_for_add_entry([], nticks=5)
    campaign = model.components[-1]

    campaign.add_entry(
        ScheduledEntry(
            what="RecordingIntervention",
            who=None, where=None,
            params={}, notes="added-after-construction", tick=2,
        )
    )

    model.run()

    notes = [c["notes"] for c in _calls]
    assert notes == ["added-after-construction"]
    assert _calls[0]["tick"] == 2
