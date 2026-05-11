# laser.cohorts

`laser.cohorts` provides composable, cohort-based (compartmental/MPM) disease models built on the [LASER](https://github.com/laser-base) simulation framework.

A *cohort model* tracks counts of individuals in each epidemiological state rather than individual agents. Each node in a spatial scenario holds an independent set of compartments; transmission and progression are applied stochastically within each node per simulation tick.

Eight standard model structures are available as ready-to-use presets — from the minimal two-compartment SI model through the four-compartment SEIRS model with waning immunity — and all can be assembled from the same library of composable components.

---

## Installation

```bash
uv pip install laser.cohorts
```

For local development from source:

```bash
git clone https://github.com/InstituteforDiseaseModeling/laser-cohorts.git
cd laser-cohorts
uv venv --python 3.13 .venv
source .venv/bin/activate  # macOS / Linux
uv pip install -e ".[dev]"
```

---

## Quick start

The example below builds and runs a 9-node SIR model for five years, then reads the final compartment counts.

```python
from laser.core import PropertySet
from laser.core.utils import grid
from laser.generic.utils import ValuesMap
from laser.cohorts import Model
import laser.cohorts.SIR as SIR

# 3×3 spatial grid; seed 10 infectious individuals per node
scenario = grid(M=3, N=3)
scenario.S -= 10
scenario.I += 10

params = PropertySet({
    "nticks": 5 * 365,
    "beta": 1.5 / 7.0,   # ~1.5 new infections per infectious individual per week
    "gamma": 1.0 / 7.0,  # average 7-day infectious period
})

model = Model(scenario, params)

betas = ValuesMap.from_scalar(params.beta, params.nticks, len(scenario))
gammas = ValuesMap.from_scalar(params.gamma, params.nticks, len(scenario))

model.components = [
    SIR.Susceptible(model),
    SIR.Infectious(model, gamma=gammas),
    SIR.Recovered(model),
    SIR.Transmission(model, beta=betas),
]

model.run()

# Access compartment time-series by name
print(model.states.S[-1])  # final susceptible counts per node
print(model.states.I[-1])  # final infectious counts per node
print(model.states.R[-1])  # final recovered counts per node
```

### Model presets

Each preset is a thin module that re-exports the correct combination of components under the conventional names `Susceptible`, `Exposed` (where applicable), `Infectious`, `Recovered` (where applicable), and `Transmission`.

| Preset | Module | Compartments |
|--------|--------|--------------|
| SI | `laser.cohorts.SI` | S, I |
| SIR | `laser.cohorts.SIR` | S, I, R |
| SIS | `laser.cohorts.SIS` | S, I (recovery → S) |
| SIRS | `laser.cohorts.SIRS` | S, I, R (waning → S) |
| SEI | `laser.cohorts.SEI` | S, E, I |
| SEIR | `laser.cohorts.SEIR` | S, E, I, R |
| SEIS | `laser.cohorts.SEIS` | S, E, I (recovery → S) |
| SEIRS | `laser.cohorts.SEIRS` | S, E, I, R (waning → S) |

### Composing custom models

Models are assembled from individual component classes, each responsible for one compartment or transition rule. Combine any subset of the primitives below:

- **Compartment components** — `Susceptible`, `Exposed`, `Infectious`, `Recovered`
- **Transition components** — `InfectiousToRecovered`, `InfectiousToSusceptible`, `RecoveredToSusceptible`
- **Transmission components** — `TransmissionSI` (S → I), `TransmissionSE` (S → E)

```python
from laser.cohorts import (
    Model, Susceptible, Exposed,
    InfectiousToRecovered, Recovered, RecoveredToSusceptible,
    TransmissionSE,
)

# SEIRS model assembled from primitives
model.components = [
    Susceptible(model),
    Exposed(model, sigma=sigmas),
    InfectiousToRecovered(model, gamma=gammas),
    Recovered(model),
    RecoveredToSusceptible(model, gamma=waning),
    TransmissionSE(model, beta=betas),
]
```

---

## Key concepts

**`Model`** — orchestrates the scenario, component list, and tick loop. Assigning `model.components` allocates the state array and node property arrays and calls each component's `setup()`. Calling `model.run()` then steps through `start_step → step → end_step` for every tick.

**`StateArray`** — a `numpy.ndarray` subclass that exposes named compartment slices as attributes (`states.S`, `states.I`, `states.R`, …). Full NumPy indexing still works (`states[0]`, `states[-1]`, etc.).

**`ValuesMap`** — a per-tick, per-node parameter map from `laser.generic`. Use `ValuesMap.from_scalar(value, nticks, n_nodes)` for homogeneous, time-constant parameters.

---

## Learn more

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } __API reference__

    ---

    Full details on all classes, components, and functions.

    [:octicons-arrow-right-24: API reference](reference/laser/cohorts/index.md)

</div>

{%
    include-markdown "../AUTHORS.md"
%}
