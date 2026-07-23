# Components

`laser.cohorts` models are assembled from small, single-responsibility component classes. Each class handles one compartment or one transition rule. Combining them builds the disease model you need.

---

## Component protocol

Every component implements the same interface:

| Method / property | Called by | Purpose |
|---|---|---|
| `states` | `model.components =` | Declare which compartment names this component uses |
| `properties` | `model.components =` | Declare per-tick, per-node output arrays |
| `setup()` | `model.components =` | Seed initial conditions from `model.scenario` |
| `start_step(tick)` | `model.run()` | Pre-step hook (rarely used) |
| `step(tick)` | `model.run()` | Apply the component's transition logic |
| `end_step(tick)` | `model.run()` | Post-step hook (rarely used) |

All six methods must exist. Components that don't need a hook implement it as a no-op.

`states` and `properties` determine what gets allocated when you assign `model.components`. The model collects the union of all declared state names and node properties, de-duplicates them, and allocates one `StateArray` and the required node property arrays before calling any `setup()`.

---

## Compartment components

These components own a compartment — they read initial conditions from `model.scenario` in `setup()` and otherwise let the carry-forward mechanism keep the values alive across ticks.

### `Susceptible`

Declares state `S`. Seeds `states.S[0]` from `scenario["S"]` at setup.

```python
Susceptible(model)
```

### `Exposed`

Declares states `E` and `I`. Seeds `states.E[0]` from `scenario["E"]` at setup. Each tick draws newly infectious individuals from a binomial and moves them from E to I, recording the flow in `nodes.newly_infectious`.

```python
Exposed(model, r_progression=r_progressions)
```

`r_progression` is a `ValuesMap` of shape `(nticks, n_nodes)` giving the per-tick, per-node rate of progression from E to I. Higher values mean shorter latent periods.

### `Infectious`

Declares state `I`. Seeds `states.I[0]` from `scenario["I"]`. No transition logic — used in models where I is a terminal state (SI) or where a separate transition component handles recovery.

```python
Infectious(model)
```

### `Recovered`

Declares state `R`. Seeds `states.R[0]` from `scenario["R"]`. No transition logic — used in SIR-type models where R is absorbing, or with `RecoveredToSusceptible` for waning immunity.

```python
Recovered(model)
```

---

## Transition components

These components move individuals between existing compartments each tick. They extend one of the compartment components and override `step()`.

### `InfectiousToRecovered`

Moves individuals from I to R. Used in SIR, SEIR, and their variants. Records the daily flow in `nodes.newly_recovered`.

```python
InfectiousToRecovered(model, r_recovery=r_recoveries)
```

In the SIR preset this class is re-exported as `SIR.Infectious` — it replaces the plain `Infectious` component and handles both the I compartment and the I→R transition.

### `InfectiousToSusceptible`

Moves individuals from I back to S (no lasting immunity). Used in SIS and SEIS models. Records the flow in `nodes.newly_susceptible`.

```python
InfectiousToSusceptible(model, r_recovery=r_recoveries)
```

### `RecoveredToSusceptible`

Moves individuals from R back to S (waning immunity). Used in SIRS and SEIRS models. Records the flow in `nodes.newly_susceptible`.

```python
RecoveredToSusceptible(model, r_waning=r_wanings)
```

!!! note
    `InfectiousToSusceptible` and `RecoveredToSusceptible` both declare
    `newly_susceptible` as a node property. When both are present in the same
    model the model de-duplicates the declaration and allocates one shared array;
    both components accumulate into it.

---

## Transmission components

Transmission components compute the force of infection each tick, draw newly infected individuals from a binomial, and move them from S into a destination compartment. They also apply inter-node mixing via `model.network`.

### Force of infection

The raw per-node force of infection is:

$$\lambda_i = \beta_i \cdot s_i \cdot \frac{I_i}{N_i}$$

where $s_i$ is the seasonality factor (1.0 by default), $I_i$ is the current infectious count, and $N_i$ is the total population. This is then mixed across nodes:

$$\lambda_i \mathrel{+}= \sum_j w_{ji} \lambda_j - \sum_j w_{ij} \lambda_i$$

where $w_{ij}$ is `model.network[i, j]`. Finally $\lambda$ is converted from a rate to a per-tick probability: $p = 1 - e^{-\lambda}$.

The raw $\lambda$ values are stored in `nodes.force_of_infection` each tick.

### `TransmissionSI`

S → I directly. Used in SI, SIR, SIS, and SIRS models.

```python
TransmissionSI(model, beta=betas)
TransmissionSI(model, beta=betas, seasonality=seasonal_factors)
```

Records newly infected individuals in `nodes.newly_infectious`.

### `TransmissionSE`

S → E (latent period). Used in SEI, SEIR, SEIS, and SEIRS models.

```python
TransmissionSE(model, beta=betas)
TransmissionSE(model, beta=betas, seasonality=seasonal_factors)
```

Records newly exposed individuals in `nodes.newly_infected`.

---

## Rate parameters and `ValuesMap`

All transition rates (`r_recovery`, `r_progression`, `r_waning`) and `beta` are expressed as **per-tick rates**, not probabilities. The conversion to a per-tick probability happens internally:

$$p = 1 - e^{-r}$$

This means rates are additive (you can sum them) and do not depend on tick length. For a 7-day infectious period with daily ticks, use `r_recovery = 1/7`.

Parameters are passed as `ValuesMap` objects — per-tick, per-node arrays of shape `(nticks, n_nodes)`.

```python
from laser.generic.utils import ValuesMap

# Uniform, time-constant rate across all nodes
r_recoveries = ValuesMap.from_scalar(1/7, params.nticks, len(scenario))

# Spatially heterogeneous rate (one value per node, constant over time)
node_rates = np.array([0.1, 0.15, 0.2, ...])   # shape (n_nodes,)
r_recoveries = ValuesMap.from_scalar(node_rates, params.nticks, len(scenario))
```

---

## Seasonality

`TransmissionSI` and `TransmissionSE` accept an optional `seasonality` parameter — a `ValuesMap` or 2-D array of shape `(nticks, n_nodes)` that multiplies `beta` each tick. A value of 1.0 (the default) means no seasonal effect.

```python
import numpy as np
from laser.generic.utils import ValuesMap

# Sinusoidal seasonality peaking at tick 0 (e.g. winter peak)
t = np.arange(params.nticks)
amplitude = 1.0 + 0.3 * np.cos(2 * np.pi * t / 365)   # shape (nticks,)
# Broadcast to all nodes
seasonal = np.broadcast_to(amplitude[:, None], (params.nticks, len(scenario)))
seasonal_map = ValuesMap.from_scalar(1.0, params.nticks, len(scenario))
seasonal_map[:] = seasonal

model.components = [
    ...,
    SIR.Transmission(model, beta=betas, seasonality=seasonal_map),
]
```

---

## Vital dynamics

### `NonDiseaseMortality`

Applies background mortality to one or more compartments each tick. Individuals are drawn from a binomial and removed. Deaths are accumulated in `nodes.non_disease_mortality`.

```python
from laser.cohorts import NonDiseaseMortality

# Apply to all states
ndm = NonDiseaseMortality(model, r_mortality=1 / (70 * 365))

# Restrict to specific states
ndm = NonDiseaseMortality(model, r_mortality=1 / (70 * 365), states=["S", "R"])
```

`r_mortality` can be a scalar, a `ValuesMap`, or a 2-D ndarray of shape `(nticks, n_nodes)`. `states` accepts any iterable of state names; `None` (the default) targets every state.

### `ConstantPopBirths`

Reads the deaths recorded by `NonDiseaseMortality` and adds the same count back into `S`, keeping total population constant. Must be ordered *after* `NonDiseaseMortality` in the component list.

```python
from laser.cohorts import NonDiseaseMortality, ConstantPopBirths

ndm = NonDiseaseMortality(model, r_mortality=1 / (70 * 365))
cpb = ConstantPopBirths(model)

model.components = [
    SIR.Susceptible(model),
    SIR.Infectious(model, r_recovery=r_recoveries),
    SIR.Recovered(model),
    SIR.Transmission(model, beta=betas),
    ndm,
    cpb,   # must follow ndm
]
```

---

## Routine immunization

### `RoutineImmunization`

`RoutineImmunization` is a component (added to `model.components`) that represents an ongoing baseline vaccination programme — for example, infant immunization that runs continuously through the simulation rather than firing as a one-off campaign.  It declares the `V` compartment, periodically moves susceptibles from `S` into `V`, and accumulates per-node counts on `nodes.ri_vaccinated`.

> **`RoutineImmunization` (component) vs `Vaccination` (intervention).**
> The campaign-driven `Vaccination` *intervention* (under `laser.cohorts.interventions`) fires on the specific ticks listed in a `Campaign` schedule and uses a direct binomial draw at the supplied `coverage` probability.  `RoutineImmunization` is a *component* that fires continuously on a fixed period and converts an annual coverage rate into the corresponding per-firing Poisson mean.  Use the intervention for discrete campaigns; use the component for routine baseline coverage.

```python
from laser.cohorts import RoutineImmunization

ri = RoutineImmunization(
    model,
    coverage=0.8,             # annual coverage rate r in [0, 1]
    eligible_fraction=1/70,   # share of S that's eligible (e.g. infants)
    period=30,                # fire every 30 ticks
)

model.components = [
    SIR.Susceptible(model),
    SIR.Infectious(model, r_recovery=r_recoveries),
    SIR.Recovered(model),
    SIR.Transmission(model, beta=betas),
    ri,
]
```

**Mathematical model.**  Given an annual coverage rate `r` (fraction of the eligible cohort the programme aims to vaccinate per year) and an eligible-fraction `ef` (the share of `S` that is age-eligible or otherwise reachable), the daily fraction of susceptibles vaccinated is

    f_daily = r * ef / 365

If the component fires every `period` ticks instead of every tick, each firing must cover the cohort accumulated over the preceding `period` days, so the per-firing fraction is

    f_period = period * r * ef / 365

On each firing tick the component draws

    draws  =  min( poisson( f_period * S ),  S )

per node, subtracts `draws` from `S[tick + 1]`, adds it to `V[tick + 1]`, and records the per-node count in `nodes.ri_vaccinated[tick]`.  The cap at `S` is what makes the model safe for large `f_period` (e.g. `r = ef = 1` and `period = 365`, where the Poisson mean equals `S` itself).

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `coverage` | `float` in `[0, 1]` | required | Annual coverage rate `r`. |
| `eligible_fraction` | `float` in `[0, 1]` | `1.0` | Share of `S` that is eligible for vaccination (e.g. ~`1/70` for infant-only programmes). |
| `period` | positive `int` | `1` | Tick period between firings; with `period=1` the component fires every tick. |

Out-of-range coverage or eligible fraction, or a non-positive / non-integer period, raises `ValueError` at construction time.

**Outputs.**

- `model.states.V` — the running vaccinated count per node, per tick.
- `model.nodes.ri_vaccinated` — the per-firing-tick count of new vaccinations per node; non-zero only on multiples of `period`.

---

## Migration

`Migration` moves individuals between nodes each tick using a 3-D routing tensor.

```python
from laser.cohorts import Migration
from laser.cohorts.utils import static_routing
import numpy as np

# Define a 2-D connectivity matrix (nnodes × nnodes)
n = len(scenario)
routing_2d = np.zeros((n, n), dtype=np.float32)
routing_2d[0, 1] = 0.5   # 50% of node 0 emigrants go to node 1
routing_2d[0, 2] = 0.5   # the other 50% go to node 2

# Promote to 3-D using broadcast_to (zero-copy, read-only view)
routing_3d = static_routing(routing_2d, params.nticks)

migration = Migration(model, r_migration=0.01, routing=routing_3d)
model.components = [..., migration]
```

`r_migration` is a scalar emigration rate applied to every state in every node each tick. `routing` is `(nticks, n_nodes, n_nodes)` — rows are source nodes, columns are destinations. Rows are normalized internally; zero-sum rows mean no emigration from that node.

For time-varying connectivity, build a real `(nticks, n, n)` array and set different slices for different periods. `static_routing(routing_2d, nticks)` is a convenience wrapper around `np.broadcast_to` for the common case where connectivity doesn't change over time.

Population is exactly conserved across migration: the last destination in the sequential-binomial decomposition absorbs any rounding remainder.
