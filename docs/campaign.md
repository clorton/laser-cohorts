# Campaign and interventions

A `Campaign` is a model component that reads a schedule of interventions and dispatches them at the right ticks, nodes, and compartment states. It is added to `model.components` alongside the epidemiological components.

---

## Quick example

```python
from laser.cohorts import Campaign
from laser.cohorts.interventions import Vaccination

Campaign.register(Vaccination)

schedule = [
    {"who": ["S"], "what": "Vaccination", "when": 30,
     "where": "*", "parameters": {"coverage": 0.8}, "notes": "round 1"},
    {"who": ["S"], "what": "Vaccination", "when": [180, 365, 545],
     "where": "*", "parameters": {"coverage": 0.6}, "notes": "boosters"},
]

campaign = Campaign(model, schedule)
model.components = [..., campaign]
model.run()
```

---

## Schedule entry fields

Each entry is a dict with six fields:

| Field | Required? | Type | Meaning |
|---|---|---|---|
| `who` | **required** | `"*"` or `list[str]` | Compartment states to target. `"*"` means all states; `["S", "R"]` restricts to those two. |
| `what` | **required** | `str` | Name of a registered `Intervention` subclass. |
| `when` | optional (default `"*"`) | see below | When to fire. |
| `where` | **required** | `"*"`, `int`, or `list[int]` | Node IDs to target. `"*"` means all nodes. |
| `parameters` | optional (default `{}`) | `dict` | Arbitrary key/value pairs forwarded to `apply()`. |
| `notes` | optional (default `""`) | `str` | Free-text annotation forwarded to `apply()`. |

!!! note
    `who` and `where` are required for every entry — omitting them raises
    `ValueError`. Use `"*"` explicitly to target all states or all nodes;
    the Campaign deliberately does **not** silently default these fields.

### `when` variants

| Value | Behaviour |
|---|---|
| `"*"` | Fires on every tick. |
| `30` | Fires once on tick 30. |
| `[30, 60, 90]` | Fires once on each listed tick. |
| `"2020-03-15"` | Fires on the tick corresponding to that date; requires `start_date`. |
| `["2020-03-15", "2020-06-01"]` | Fires once on each listed date; requires `start_date`. |

Integer ticks and date strings cannot be mixed in the same schedule — neither across entries nor within a single list. Dates earlier than `start_date` raise `ValueError`.

```python
# Date-based schedule
campaign = Campaign(model, schedule, start_date="2020-01-01")
```

Out-of-range ticks (beyond `params.nticks`) are silently skipped — the model simply never reaches them.

---

## Loading sources

`Campaign` accepts five source formats:

=== "dict (single entry)"

    ```python
    entry = {"who": "*", "what": "Vaccination", "when": 0,
             "where": "*", "parameters": {"coverage": 0.9}, "notes": ""}
    campaign = Campaign(model, entry)
    ```

=== "list"

    ```python
    schedule = [
        {"who": ["S"], "what": "Vaccination", "when": 30,  "where": "*", "parameters": {"coverage": 0.8}, "notes": ""},
        {"who": ["S"], "what": "Vaccination", "when": 180, "where": [0, 1], "parameters": {"coverage": 0.7}, "notes": ""},
    ]
    campaign = Campaign(model, schedule)
    ```

=== "JSON file"

    ```json
    [
      {"who": ["S"], "what": "Vaccination", "when": 30,
       "where": "*", "parameters": {"coverage": 0.8}, "notes": ""},
      {"who": ["S"], "what": "Vaccination", "when": [180, 365],
       "where": "*", "parameters": {"coverage": 0.6}, "notes": "boosters"}
    ]
    ```

    ```python
    campaign = Campaign(model, "schedule.json")
    ```

=== "YAML file"

    ```yaml
    - who: [S]
      what: Vaccination
      when: 30
      where: "*"
      parameters: {coverage: 0.8}
      notes: round 1

    - who: [S]
      what: Vaccination
      when: [180, 365]
      where: "*"
      parameters: {coverage: 0.6}
      notes: boosters
    ```

    ```python
    campaign = Campaign(model, "schedule.yaml")
    ```

    Both `.yaml` and `.yml` extensions are accepted. A single top-level
    mapping is promoted to a one-entry schedule, mirroring the JSON loader.
    Quote any literal `"*"` value so it isn't read as a YAML alias anchor,
    and quote `"YYYY-MM-DD"` `when` values to keep them as strings rather
    than YAML-native dates.

=== "CSV file"

    ```csv
    who,what,when,where,parameters,notes
    ["S"],Vaccination,30,*,"{""coverage"": 0.8}",round 1
    ["S"],Vaccination,"[180, 365]",*,"{""coverage"": 0.6}",boosters
    ```

    ```python
    campaign = Campaign(model, "schedule.csv")
    ```

    In CSV, list-valued fields (`who`, `where`, `when`) are JSON-encoded strings.
    `parameters` is a JSON object string.

Three equivalent reference files — `campaign_sample.json`, `campaign_sample.yaml`, and `campaign_sample.csv` — live in `tests/data/` and serve as worked examples for each file format.

---

## Built-in interventions

### `Vaccination`

Moves a binomial-drawn fraction of each targeted state into a dedicated `V` (vaccinated) compartment. Declares the `V` state and a `newly_vaccinated` node property, so the model allocates both automatically.

```python
from laser.cohorts.interventions import Vaccination

Campaign.register(Vaccination)

schedule = [
    {"who": ["S"],    "what": "Vaccination", "when": 30,
     "where": "*",   "parameters": {"coverage": 0.8}, "notes": ""},
]
```

**Parameters:**

| Key | Type | Default | Meaning |
|---|---|---|---|
| `coverage` | `float` | `0.0` | Probability in [0, 1] that any targeted individual is vaccinated this tick. |

`coverage` must be in `[0, 1]`; values outside this range raise `ValueError` at the tick when the intervention fires.

**Compartment transitions:**

```
S (or other targeted states)  →  V
```

After the campaign fires, `model.nodes.newly_vaccinated[tick]` holds the per-node count of newly vaccinated individuals.

---

## Writing a custom intervention

Subclass `Intervention` and implement `apply()`. Override `states` and/or `properties` if your intervention needs new compartments or output arrays.

```python
from laser.cohorts.campaign import Intervention
import numpy as np

class SeedInfection(Intervention):
    """Add a fixed number of infected individuals to targeted nodes."""

    def apply(self, tick, who, where, params, notes):
        count = int(params.get("count", 1))
        nnodes = len(self.model.scenario)
        target_nodes = where if where is not None else list(range(nnodes))

        states_next = self.model.states[tick + 1]
        I_idx = self.model.states.get_state_index("I")
        if I_idx is None:
            return

        for node in target_nodes:
            available = int(states_next[I_idx, node])
            # Only seed if there are susceptibles to draw from (tracked elsewhere)
            states_next[I_idx, node] = available + count
```

Register the class before assigning `model.components`:

```python
Campaign.register(SeedInfection)

schedule = [
    {"who": "*", "what": "SeedInfection", "when": 0,
     "where": [3], "parameters": {"count": 5}, "notes": "re-introduction"},
]
campaign = Campaign(model, schedule)
model.components = [..., campaign]
```

### Declaring new states

If your intervention creates a new compartment, declare it via `states`:

```python
class Quarantined(Intervention):
    @property
    def states(self):
        return ["Q"]

    def apply(self, tick, who, where, params, notes):
        ...   # move individuals into states.Q
```

### Declaring new node properties

If your intervention records per-tick output, declare it via `properties`:

```python
from laser.cohorts.utils import PropertyType
import numpy as np

class Quarantined(Intervention):
    @property
    def properties(self) -> list[PropertyType]:
        return [("newly_quarantined", self.model.params.nticks, np.int32, 0)]

    def apply(self, tick, who, where, params, notes):
        ...
        self.model.nodes.newly_quarantined[tick] += drawn
```

`Campaign.states` and `Campaign.properties` automatically aggregate these declarations from all interventions referenced in the schedule, so the model allocates everything before `setup()` runs.

---

## Registration

Intervention classes must be registered before `model.components` is assigned. Registration uses the class `__name__` as the key; that name must match the `what` field in the schedule.

```python
Campaign.register(Vaccination)       # key = "Vaccination"
Campaign.register(SeedInfection)     # key = "SeedInfection"
```

Registration is class-level and persistent for the lifetime of the Python process. If you run multiple models in the same session you only need to register once.

An unregistered name in the schedule raises `ValueError` at construction time — `Campaign._validate` checks the registry the moment the schedule is loaded, so the error surfaces immediately rather than when the offending tick is dispatched.

---

## Adding interventions at runtime — `Campaign.add_entry`

The schedule passed to `Campaign(...)` defines every dispatch known at construction time, but `Campaign` can be extended *during* a run.  Use `Campaign.add_entry(entry)` to schedule a follow-up intervention from inside another intervention's `apply()` — typical pattern for reactive surveillance, cascading interventions, or any scenario where the right tick to fire on is only known after some condition is observed.

### The `ScheduleEntry` dataclass

`ScheduleEntry` is the normalised, fully-parsed form of a schedule row.  After `Campaign` loads and validates a schedule, every raw entry is expanded into one or more `ScheduleEntry` instances (one per resolved tick).  You build one explicitly when calling `add_entry`:

```python
from laser.cohorts import Campaign, ScheduleEntry

ScheduleEntry(
    what="Vaccination",        # registered intervention name
    who=["S"],                  # or None for all states
    where=[0],                  # or None for all nodes
    params={"coverage": 0.7},
    notes="reactive round triggered at tick 154",
    tick=155,                   # absolute tick to fire on, or None for every tick
)
```

### Calling `add_entry` from inside `apply()`

```python
class Surveillance(Intervention):
    def apply(self, tick, who, where, params, notes):
        # 1. Detect — count recent cases, decide which nodes to flag
        ...
        if not alarms:
            return

        # 2. Locate the Campaign we belong to
        campaign = next(c for c in self.model.components if isinstance(c, Campaign))

        # 3. Schedule a follow-up Vaccination round for tick + 1
        campaign.add_entry(
            ScheduleEntry(
                what="Vaccination",
                who=["S"],
                where=sorted(alarms),
                params={"coverage": 0.7},
                notes=f"reactive vaccination for {alarms}",
                tick=tick + 1,
            )
        )
```

### Semantics

- **Validation.** `add_entry` validates the entry's `what` against the same registry as construction-time validation: an unregistered name raises `ValueError`, and a non-`ScheduleEntry` argument raises `TypeError`.
- **Routing.** Entries with `tick=None` go into the every-tick bucket and fire on every subsequent tick.  Entries with a concrete `tick` go into `Campaign._at_tick[tick]` and fire on that tick.
- **Visibility.** `Campaign.step` re-reads its dispatch list at the start of every tick, so an entry added during dispatch at tick `T` is visible from tick `T+1` onward.  An entry added for the current tick or any tick already in the past silently does nothing.

For a fully worked example — monthly surveillance that reactively vaccinates alarmed nodes (and optionally their network neighbours) — see `nb_19_reactive_campaign.ipynb`.
