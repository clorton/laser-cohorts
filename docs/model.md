# Model

`Model` is the central object in `laser.cohorts`. It holds the scenario, parameters, and component list, allocates all simulation arrays, and runs the tick loop.

---

## Lifecycle

Every simulation goes through three phases:

```
Model(scenario, params)          # 1. construct
    ↓
model.components = [...]         # 2. allocate states, call setup()
    ↓
model.run()                      # 3. execute the tick loop
```

### 1. Construct

```python
from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model

scenario = grid(M=3, N=3)   # 3×3 spatial grid — 9 nodes
scenario["S"] = 990
scenario["I"] = 10
scenario["R"] = 0

params = PropertySet({"nticks": 365})

model = Model(scenario, params)
```

`scenario` is a GeoDataFrame where each row is a node. Any columns you add (``S``, ``I``, ``R``, …) become the initial conditions read by each component's ``setup()``.

`params` is a ``PropertySet`` that must include ``nticks``. Components may also read arbitrary keys from ``params`` (e.g. ``params.beta``) at construction time.

### 2. Assign components

```python
import laser.cohorts.SIR as SIR
from laser.generic.utils import ValuesMap

nticks = params.nticks
n = len(scenario)

betas      = ValuesMap.from_scalar(0.3,   nticks, n)
r_recoveries = ValuesMap.from_scalar(0.1, nticks, n)

model.components = [
    SIR.Susceptible(model),
    SIR.Infectious(model, r_recovery=r_recoveries),
    SIR.Recovered(model),
    SIR.Transmission(model, beta=betas),
]
```

Assigning `model.components` triggers three things in order:

1. **State collection** — each component's `states` property is queried; the union of all unique state names determines the compartments allocated.
2. **Array allocation** — `model.states` (a `StateArray` of shape `(nticks+1, n_states, n_nodes)`) and all node property arrays are created.
3. **Setup** — each component's `setup()` is called, seeding initial conditions from `scenario`.

Attempting to read `model.states` before assigning `model.components` raises `AttributeError`.

### 3. Run

```python
model.run()
```

Steps the simulation for `params.nticks` ticks. See [The tick loop](#the-tick-loop) below.

---

## The tick loop

Each tick executes in this order:

```
carry-forward:  states[tick+1][mask] = states[tick][mask]
    ↓
for each component: start_step(tick)
    ↓
for each component: step(tick)
    ↓
for each component: end_step(tick)
```

The carry-forward runs *before* any component sees the tick. Components read and write `states[tick+1]` during their `step()`, starting from the carried-forward values.

---

## Accessing results

`model.states` is a `StateArray` — a NumPy array subclass that exposes named compartment slices as attributes.

```python
# Time-series for each node (shape: nticks+1, n_nodes)
model.states.S          # all ticks, all nodes
model.states.I[0]       # tick 0, all nodes (initial conditions)
model.states.R[-1]      # last tick, all nodes (final counts)

# Full snapshot at a given tick (shape: n_states, n_nodes)
model.states[10]        # plain ndarray, tick 10

# Scalar for a specific tick and node
int(model.states.S[30, 2])   # S at tick 30, node 2
```

Node properties (e.g. per-tick flow counts recorded by components) are on `model.nodes`:

```python
model.nodes.newly_infectious     # shape: (nticks, n_nodes)
model.nodes.force_of_infection   # shape: (nticks, n_nodes)
```

---

## Selective carry-forward

By default every compartment state is carried from tick `t` to tick `t+1`. Pass `carry_forward_states` to restrict this to a subset:

```python
model = Model(scenario, params, carry_forward_states=["S", "I"])
```

States not in the list start at zero on each tick — they accumulate only what components write into `states[tick+1]` during that tick's `step()`. Passing an unknown state name raises `ValueError` when `model.components` is assigned.

Selective carry-forward is most useful when a custom component completely replaces a state each tick and carry-forward would interfere with that logic.

---

## Inter-node mixing

`model.network` is a 2-D array of shape `(n_nodes, n_nodes)` whose entry `[i, j]` is the connectivity weight from node `i` to node `j`. The transmission components use this matrix to spread force of infection across nodes.

```python
import numpy as np

# Uniform weak coupling between all node pairs
n = len(scenario)
model.network = np.full((n, n), 0.01, dtype=np.float32)
np.fill_diagonal(model.network, 0.0)   # no self-coupling
```

The default is an all-zero matrix (no inter-node coupling).

!!! note
    `model.network` must be assigned *before* `model.components`, because the
    transmission component's `setup()` does not cache the network array — it
    reads `model.network` live during each `step()`.
