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
| `parameters` | optional (default `{}`) | `dict` | Arbitrary key/value pairs forwarded to `execute()`. |
| `notes` | optional (default `""`) | `str` | Free-text annotation forwarded to `execute()`. |

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

`Campaign` accepts four source formats:

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

Subclass `Intervention` and implement `execute()`. Override `states` and/or `properties` if your intervention needs new compartments or output arrays.

```python
from laser.cohorts.campaign import Intervention
import numpy as np

class SeedInfection(Intervention):
    """Add a fixed number of infected individuals to targeted nodes."""

    def execute(self, tick, who, where, params, notes):
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

    def execute(self, tick, who, where, params, notes):
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

    def execute(self, tick, who, where, params, notes):
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

An unregistered name in the schedule raises `KeyError` when the scheduled tick is reached during `model.run()`.
