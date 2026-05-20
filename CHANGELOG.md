# Changelog

## Unreleased

### Added (Campaign.add_entry — runtime intervention scheduling)
- `src/laser/cohorts/campaign.py`: renamed module-private `_ScheduleEntry` to public `ScheduleEntry`; the dataclass is now part of the documented surface
- `Campaign.add_entry(entry: ScheduleEntry) -> None`: new public method that validates the entry's `what` is registered and routes it to either `Campaign._every_tick` (if `entry.tick is None`) or the appropriate `Campaign._at_tick[entry.tick]` bucket; raises `TypeError` on wrong type, `ValueError` on unregistered intervention
- `src/laser/cohorts/__init__.py`: exports `ScheduleEntry` alongside `Campaign` and `Intervention`
- `docs/notebooks/nb_19_reactive_campaign.ipynb`: `Surveillance.execute` now schedules reactive vaccination via `campaign.add_entry(ScheduleEntry(...))` instead of reaching into `Campaign._at_tick` directly; markdown and discussion updated to reflect the new public API
- `tests/test_campaign.py`: added 5 tests covering tick-specific routing, every-tick routing, dispatch from inside another intervention's `execute()`, `TypeError` for non-`ScheduleEntry` input, and `ValueError` for unregistered `what`

### Changed (Campaign — parsed entries are dataclass instances)
- `src/laser/cohorts/campaign.py`: introduced `_ScheduleEntry`, a module-level `@dataclass` with fields `what`, `who`, `where`, `params`, `notes`, and `tick`
- `Campaign._parse_entries` now returns `list[_ScheduleEntry]` instead of `list[dict]` — same content, attribute access instead of string-keyed lookup
- `Campaign.setup`, `Campaign.step`, `Campaign.properties`, and `Campaign.states` updated to read fields via `entry.what` / `entry.who` / `entry.where` / `entry.params` / `entry.notes` / `entry.tick`
- `Campaign._every_tick` and `Campaign._at_tick` type annotations updated to `list[_ScheduleEntry]` / `dict[int, list[_ScheduleEntry]]`
- `properties` and `states` also dropped the dead `name not in self._registry` guard — validation now ensures all `what` names are registered

### Changed (Campaign — registered-intervention check moved to validation)
- `src/laser/cohorts/campaign.py` `Campaign._validate`: added a per-entry check that every `what` value names an intervention class that has already been registered via `Campaign.register`; unregistered names now raise `ValueError` at construction time with a message that points to the missing registration call
- `Campaign.step`: removed the runtime `if name not in self._registry: raise KeyError(...)` guard — the lookup is now unconditional because validation has already ensured every name resolves; method docstring updated accordingly
- `Campaign.__init__` Raises section adds the new unregistered-`what` `ValueError`
- `tests/test_campaign.py`: renamed `test_unknown_intervention_name_raises_key_error` → `test_unknown_intervention_name_raises_value_error_at_construction`; the test now asserts the `ValueError` fires from `Campaign(...)` rather than from `model.run()`

### Added (Campaign — explicit `_validate` stage)
- `src/laser/cohorts/campaign.py`: new `Campaign._validate(raw)` method runs between `_load` and `_parse_entries`. It checks (1) every entry has required `who`, `what`, and `where` fields; (2) each `who`/`where`/`when` value matches the documented per-field grammar; (3) `when` lists are homogeneous (no mixing of ints and dates within a list); (4) no cross-entry mixing of date and integer `when` values; (5) `start_date` is provided when dates are used and no date precedes it
- new module-level validators `_validate_who_value`, `_validate_where_value`, `_validate_when_value` and predicates `_is_int_like` / `_is_date_like` carry the per-value grammar rules
- `Campaign._parse_entries` is now pure conversion — all the cross-entry and required-field checks moved to `_validate`
- `_normalize_when` simplified to pure conversion (no more mixed-list, missing-`start_date`, or date-before-start checks); these errors are now surfaced earlier with the same messages
- Missing `what` now raises `ValueError` at construction time instead of later `KeyError` inside `step()`
- Invalid value shapes (e.g. `who=42`, `where="abc"`, `when="notadate"`, mixed lists) now raise `ValueError` at construction with a field-specific message instead of crashing later inside `_normalize_*` or `int()`
- `tests/test_campaign.py`: added 6 new tests covering missing `what`, invalid `who` int, invalid `where` non-numeric string, invalid `when` non-date string, list with wrong element type for `who`, and list with wrong element type for `where`

### Changed (Campaign — unified CSV cell handling for `who`/`where`/`when`)
- `src/laser/cohorts/campaign.py` `_load_csv`: the field-specific `raw_who` / `raw_where` / `raw_when` blocks are replaced by a single nested `_normalize_cell(value)` helper applied uniformly to all three fields via a `for field in ("who", "where", "when")` loop; the cell grammar (empty → drop, `"*"` → keep, `"[…]"` → JSON, else → bare string) is now identical across the three
- `_normalize_who` updated to accept a bare scalar string (e.g. `who="S"`) and wrap it as `["S"]`, parallel to how `_normalize_where` and `_normalize_when` already handle single scalars; bracketed strings still parse as JSON arrays
- `tests/test_campaign.py`: added `test_who_bare_string_wrapped_as_single_element_list` (dict input) and `test_csv_who_bare_string_cell_parses_as_single_element_list` (CSV input)

### Changed (Campaign — set-based dedup of `states` and `properties`)
- `src/laser/cohorts/campaign.py` `Campaign.states` and `Campaign.properties`: replaced the manual `if X not in result: result.append(X)` dedup with a `set` accumulator and `list(result)` at the end; PropertyType tuples and state strings are both hashable so this works without any change to the public API
- docstrings updated to note that the returned-list order is now unspecified (the `Model.components` setter already deduplicates downstream, so order here is not load-bearing)

### Changed (Campaign — `who` and `where` are now required)
- `src/laser/cohorts/campaign.py` `_parse_entries`: omitting the `who` or `where` field from a schedule entry now raises `ValueError`; users must specify `"*"` explicitly to target all states or all nodes (silent defaulting hid configuration mistakes)
- `_load_csv`: a missing `who`/`where` column or an empty `who`/`where` cell is now mapped to an absent dict key so `_parse_entries` produces the same clean error, rather than the previous opaque `json.JSONDecodeError` for empty cells
- module docstring, `Campaign.__init__` Raises section, and `docs/campaign.md` updated to mark `who`/`where` as required; `docs/campaign.md` schedule-fields table now has a Required? column and an admonition spelling out the rule
- `docs/campaign.md` `when` variants table now also includes the list-of-dates form (previously stated as unsupported)
- `tests/test_campaign.py`: added 4 new tests — `test_missing_who_raises_value_error`, `test_missing_where_raises_value_error`, `test_csv_missing_who_column_raises_value_error`, `test_csv_empty_where_cell_raises_value_error`

### Changed (Campaign — extracted `_normalize_when`)
- `src/laser/cohorts/campaign.py`: extracted a module-level `_normalize_when(raw, start_date)` helper that mirrors `_normalize_who` and `_normalize_where`; it returns `list[int] | None` (`None` for `"*"`) and handles all `when` forms (int, int-string, date string, list of ints, list of dates)
- `_normalize_when` raises `ValueError` for: lists that mix date strings and non-dates, date-valued inputs missing `start_date`, and dates earlier than `start_date`
- `_parse_entries`: simplified to (1) detect cross-entry mixing of date and integer `when`, then (2) call `_normalize_when` per entry and expand returned tick lists into one parsed entry per tick
- removed the now-redundant `Campaign._date_to_tick` method (logic absorbed by `_normalize_when`)

### Changed (Campaign — list-of-dates support in `when`)
- `src/laser/cohorts/campaign.py` `_parse_entries`: lists of date strings (e.g. `["2020-01-10", "2020-02-01"]`) are now accepted in the `when` field and expanded into one firing per listed date, with each date converted to a tick offset from `start_date`
- new helper `Campaign._date_to_tick` centralises the date-to-tick conversion and raises `ValueError` if any date precedes `start_date`
- mixed dates and integer ticks within a single list (e.g. `["2020-01-10", 15]`) now raise `ValueError` early; the `has_dates` / `has_int_ticks` detection in `_parse_entries` now inspects list contents in addition to scalar entries
- module docstring and `Campaign.__init__` Raises section updated to document the new list-of-dates form and the date-before-start_date error
- **Fixed** a pre-existing inverted comparison bug in the scalar-date branch (`self._start_date < d` → `d < self._start_date`) so single-date `when` values no longer spuriously raise "is before campaign start date" for dates that actually fall after start_date
- `tests/test_campaign.py`: removed obsolete `test_when_list_of_dates_raises_value_error`; added 7 new tests covering happy-path date-list firing, out-of-range date skipping, per-firing metadata preservation, missing-start_date `ValueError`, mixed-list `ValueError`, date-before-start `ValueError`, and CSV JSON-array-of-dates round-trip; shared `_make_model_with_date_schedule` helper added

### Changed (nb_15 gravity model)
- `docs/notebooks/nb_15_england_wales_model.ipynb`: replaced manual gravity calculation with `laser.core.migration.gravity(pops, distances, k, a, b, c)`; removed manual `d_safe` div-by-zero guard and `np.fill_diagonal` call (handled internally by the library)

### Added (notebooks 15–16 — England & Wales spatial measles)
- `docs/notebooks/nb_15_england_wales_model.ipynb`: Spatial SEIR model for measles in England and Wales (1944–1964) — 954-node gravity-coupled model with seasonal forcing, vital dynamics, and external importation; saves simulated incidence, populations, and placenames to `data/` for use by nb_16
- `docs/notebooks/nb_16_ew_analysis.ipynb`: Analysis notebook — loads nb_15 outputs and compares simulated vs. observed fadeout proportions (fraction of bi-weekly periods with zero cases) as a function of city population; time-series comparison for selected cities; geographic map of persistence; summary of model strengths and limitations
- `mkdocs.yml`: added "England & Wales spatial SEIR model" (nb_15) and "England & Wales analysis" (nb_16) to the Tutorials nav after nb_14

### Added (tutorials nav and notebooks 01-03, 09)
- `mkdocs.yml`: added Tutorials section to nav with entries for all 14 notebooks
- `docs/notebooks/nb_01_si_logistic_growth.ipynb`: SI logistic growth (no vital dynamics) — verifies S+I=N, fits logistic curve with scipy.optimize.curve_fit, asserts β recovery within 5%
- `docs/notebooks/nb_02_sis_logistic_growth.ipynb`: SIS logistic growth (no vital dynamics) — shows endemic equilibrium I* = N(1 − γ/β) and extinction when R₀ < 1
- `docs/notebooks/nb_03_sir_outbreak_size.ipynb`: SIR outbreak size (Kermack–McKendrick) — solves final-size equation with brentq, sweeps R₀ values, scatter of expected vs. observed attack rate
- `docs/notebooks/nb_09_seasonality.ipynb`: Seasonality — SIR + vital dynamics with β(t) = β₀(1 + ε·cos(2πt/365)); compares sustained forced oscillations vs. damped unforced oscillations

### Added (notebooks 10–14)
- `docs/notebooks/nb_10_two_patch_correlation.ipynb`: Two-patch SIR spatial correlation (Keeling & Rohani 2002) — sweeps migration σ over 100 log-spaced values, computes post-burn Pearson correlation, fits C(σ) = σ/(ξ+σ) with scipy curve_fit to recover ξ
- `docs/notebooks/nb_11_rabies_1d_diffusion.ipynb`: Rabies 1D spatial diffusion traveling wave — 101-node nearest-neighbour SI chain seeded at node 50; spatiotemporal heatmap and measured vs. theoretical wavespeed c = 2√(D·r)
- `docs/notebooks/nb_12_ccs.ipynb`: Critical Community Size (CCS) — sweeps N from 10³ to 10⁶ with 50 replicates each; measures persistence fraction; plots inflection point vs. theoretical I*/N = 1 heuristic
- `docs/notebooks/nb_13_age_at_infection.ipynb`: Age at first infection — endemic SIR with vital dynamics; extracts newly_infectious node property; computes force of infection λ* and mean age A = 1/λ*; discusses cohort-model limitation
- `docs/notebooks/nb_14_routine_immunization.ipynb`: Routine immunization — SEIR + vital dynamics; three coverage scenarios (0, 0.5, 0.8) using built-in Vaccination intervention; compares I(t) time series and cumulative infections; notes limitation of no age structure

### Added (notebooks 4–8)
- `docs/notebooks/nb_04_si_vital_dynamics.ipynb`: SI model with vital dynamics — validates endemic equilibrium I* = N(1 − μ/β) and N(t) stability with NonDiseaseMortality + ConstantPopBirths
- `docs/notebooks/nb_05_sir_natural_periodicity.ipynb`: SIR natural periodicity and disease importation — demonstrates sustained oscillations with annual importation vs. damped oscillations without; computes theoretical period T = 2π/ω
- `docs/notebooks/nb_06_mortality.ipynb`: NonDiseaseMortality validation — 10 replicates × 4 CDR values (2, 10, 20, 40 per 1000/yr); confirms observed CDR matches input within ±0.5
- `docs/notebooks/nb_07_constant_pop.ipynb`: ConstantPopBirths validation on a 5×5 spatial grid (25 nodes); confirms N_total(t) stays within 1% of initial value over 10 years
- `docs/notebooks/nb_08_births_varying_cbr.ipynb`: three birth/death balance scenarios (constant CBR, two-node per-node CBR, declining CBR) with β = 0; demonstrates population stability and decline when births fall below deaths

### Added (documentation)
- `docs/model.md`: new page covering the Model lifecycle (construct → assign components → run), the tick loop order (carry-forward then start_step/step/end_step), `StateArray` access patterns, `carry_forward_states`, and `model.network`
- `docs/components.md`: new page covering the component protocol, all compartment/transition/transmission components with parameter names, rate-to-probability conversion, `ValuesMap` usage, seasonality, `NonDiseaseMortality`, `ConstantPopBirths`, and `Migration`
- `docs/campaign.md`: new page covering schedule entry fields, `when` variants, all four loading sources (dict/list/JSON/CSV), the `Vaccination` built-in, and a full custom intervention example with `states`/`properties` declarations
- `docs/index.md`: fixed stale parameter names (`gamma` → `r_recovery`, `sigma` → `r_progression`, `gamma` (waning) → `r_waning`) in Quick start and Composing custom models code examples; added three new "Learn more" cards linking to the new pages
- `mkdocs.yml`: added Model, Components, and Campaign & interventions entries to the nav

### Changed (Model carry_forward_states uses get_state_mask)
- `src/laser/cohorts/model.py` `Model.components` setter: replaced manual mask-building loop with `StateArray.get_state_mask()`; unknown state names in `carry_forward_states` now raise `ValueError` immediately instead of emitting a `UserWarning` and continuing
- `src/laser/cohorts/model.py`: removed now-unused `import warnings`
- `src/laser/cohorts/model.py`: updated `__init__` and `components` setter docstrings to document `ValueError` on unknown state names
- `tests/test_model.py`: renamed `test_carry_forward_unknown_state_name_emits_warning` → `test_carry_forward_unknown_state_name_raises_value_error`; replaced `warnings.catch_warnings` assertion with `pytest.raises(ValueError)`; removed `import warnings`

### Fixed (StateArray.get_state_mask)
- `src/laser/cohorts/statearray.py` `get_state_mask`: fixed `NameError` bug where the type-check error message referenced the undefined variable `state` instead of `states`; passing a non-str, non-list argument (e.g. a tuple) now raises `ValueError` as intended
- `get_state_mask`: added Google-style docstring with Args, Returns, Raises, and executable Example
- `tests/test_statearray.py` `TestGetStateMask`: 10 new tests covering single-string input, list-of-one, list-of-multiple, all-states (all True), empty list (all False), mask length, unknown name `ValueError`, unknown name in list `ValueError`, non-list/non-str input `ValueError` (the bug-fix path), and NumPy boolean indexing integration

### Added (Vaccination intervention)
- `src/laser/cohorts/interventions/vaccination.py`: implemented `Vaccination` intervention; reads `coverage` from `params` (default `0.0`, must be in `[0, 1]`); applies binomial draw to each targeted state in each targeted node and moves drawn individuals into the V compartment; accumulates per-node counts in the `newly_vaccinated` node property; raises `ValueError` if coverage is outside `[0, 1]`
- `Intervention` base class extended with `states` (default `[]`) and `properties` (default `[]`) instance properties; `Campaign.states` and `Campaign.properties` aggregate these across all unique intervention classes referenced in the schedule
- `src/laser/cohorts/interventions/__init__.py`: created with `Vaccination` export
- `tests/test_vaccination.py`: 15 tests covering `Vaccination.states`/`properties` declarations, `Campaign.states`/`Campaign.properties` aggregation, coverage=0 deterministic (nobody vaccinated), coverage=1 deterministic (all targeted), invalid coverage `ValueError`, `who` list restriction to named states, `who=None` targeting all states, `where` list restriction to named nodes, `where=None` targeting all nodes, `newly_vaccinated` count recording, `newly_vaccinated` zero on unscheduled ticks, population conservation (binomial draw), and V carry-forward across ticks

### Added (Campaign)
- `src/laser/cohorts/campaign.py`: implemented `Campaign` component and `Intervention` base class; `Campaign` loads a schedule from a `dict`, `list[dict]`, `.json` file, or `.csv` file and dispatches registered intervention classes at the specified ticks, nodes, and compartment states
- Schedule entry fields: `who` (`"*"` or list of state names), `what` (registered class name), `when` (`"*"` every tick, integer tick, **list of integer ticks**, or `"YYYY-MM-DD"` date), `where` (`"*"`, single node ID, or list of node IDs), `parameters` (arbitrary key:value dict), `notes` (free-text string)
- A list `when` such as `[30, 60, 90]` expands into one firing per listed tick; out-of-range ticks are silently skipped; list of date strings raises `ValueError`
- Date-based `when` values require a `start_date` constructor argument; integer ticks / tick-lists and date strings cannot be mixed in the same schedule (raises `ValueError`)
- `Campaign.register(cls)` classmethod registers `Intervention` subclasses using `cls.__name__` as the key; unknown names raise `KeyError` when the scheduled tick fires
- CSV `parameters`, `who` (list), `where` (list), and `when` (list) columns accept JSON-encoded strings/arrays
- `laser.cohorts.Campaign` and `laser.cohorts.Intervention` exported from `__init__.py`
- `tests/test_campaign.py`: 27 tests covering dict/list/JSON/CSV sources, `when="*"` (every tick), integer, list-of-ticks (full and partially out-of-range), and date `when`, CSV JSON array `when`, metadata forwarding across list firings, list-of-dates `ValueError`, `where`/`who` normalization, parameters/notes forwarding, multiple same-tick entries, unknown class name `KeyError`, mixed date/tick `ValueError`, missing `start_date` `ValueError`, unsupported file format `ValueError`

### Changed (Migration 3-D routing + vectorised step)
- `src/laser/cohorts/migration.py`: `routing` parameter promoted from 2-D `(nnodes, nnodes)` to 3-D `(nticks, nnodes, nnodes)`; connectivity can now vary tick-by-tick; static connectivity expressed via `np.broadcast_to(routing_2d[None], (nticks, n, n))` (read-only view, no copy)
- `Migration.__init__`: normalisation axes updated (`axis=2`); `_emigrates` is now `(nticks, nnodes)` bool; `ValueError` message now suggests `np.broadcast_to` for the 2-D→3-D conversion
- `Migration.step`: replaced nested `for i, for s` Python loop with a vectorised sequential-binomial decomposition of the multinomial — one `binomial` call per destination column, fully vectorised over `(nstates, nnodes)`; last destination absorbs remainder for exact population conservation; `routing_tick = self._routing[tick]` slices the current tick's routing slice
- `tests/test_migration.py`: added `static_routing(routing_2d, nticks)` helper that wraps `np.broadcast_to` and is used by all 10 original tests; updated all tests to pass 3-D routing; `test_migration_raises_on_wrong_routing_shape` now also verifies that a plain 2-D array is rejected; added 3 new time-varying tests: routing inactive first-half / active second-half (deterministic), alternating direction each tick (population conserved both ways), and 30-tick period rotation across three directed patterns with active disease (population conserved at every tick); total migration tests: 13

### Added (Migration)
- `laser.cohorts.Migration` exported from `__init__.py`

### Added (seasonality and network tests)
- `tests/test_seasonality.py`: 8 tests covering zero seasonality (no transmission), unit constant matching None default, doubled seasonality increasing infections, step-function first-half zeros, sinusoidal trough (T=364, tick 273), triangle-wave trough (mid-simulation), two-peak annual pattern (four trough zeros over two periods), and extreme 20000× seasonality depleting all susceptibles in the first tick
- `tests/test_networks.py`: 8 tests covering zero network isolation, full all-to-all connectivity spreading to all nodes, one-directional linear chain, hub-and-spoke two-hop spread, one-directional ring, asymmetric one-way spread with deterministic blocked reverse, symmetric w=0.5 connectivity equalising recovered fractions (within 5pp), and isolated node remaining pristine

### Added (documentation)
- `docs/index.md`: rewrote from `laser.generic` boilerplate to `laser.cohorts`-specific content — description, installation, SIR quick-start example, model preset table, key-concepts summary, and API reference link
- `mkdocs.yml`: updated `site_name`, `site_url`, `repo_name`, `repo_url` to `laser.cohorts`; simplified `nav` to only existing pages (`index.md` and `reference/`)
- `docs/customization/gen-files.py`: corrected SUMMARY.md generation to reference `laser.cohorts` instead of `laser.generic`; removed `laser.core` collection and page generation (cross-reference resolution via `preload_modules` in `mkdocs.yml` is retained)

### Added
- Google-style docstrings to all public and private methods, classes, and modules in `src/laser/cohorts/`: `__init__.py`, `model.py`, `components.py`, `statearray.py`, `utils.py`
- Module-level docstring to `src/laser/cohorts/utils.py`
- Given/when/then docstrings to all test functions in `tests/test_sir.py`, `tests/test_si.py`, `tests/test_sis.py`, `tests/test_sirs.py`, `tests/test_sei.py`, `tests/test_seir.py`, `tests/test_seirs.py`, `tests/test_seis.py`
- Args/Returns docstrings to all `run_model()` helper functions in the above test files
- Module-level documentation to `tests/test_components.py` noting it is a standalone script rather than a pytest test module

### Fixed
- `StateArray.__new__` docstring: corrected `Returns` section to Google style (removed erroneous blank line, fixed type format)
- `StateArray.state_axis` docstring: added missing `Raises` section documenting `RuntimeError`
- `StateArray.state_names` docstring: added `Returns` section
- `StateArray.get_state_index` docstring: added `Args` and `Returns` sections
- `tests/test_statearray.py` module docstring: corrected incorrect reference to `utils.py` (StateArray lives in `statearray.py`)
- `tests/test_statearray.py` `TestStateArray` class: replaced 11 single-line docstrings with Given/When/Then descriptions
- `tests/test_statearray.py` `TestConstructionPaths.test_view_casting` and `test_new_from_template`: replaced embedded code-snippet docstrings with Given/When/Then descriptions
- `tests/test_statearray.py` `TestConstructionPaths.test_from_ufunc`: added missing docstring

### Fixed (third pass — final audit items)
- `tests/test_statearray.py` fixtures `zero_data`, `sample_data`, `tsp_data`, `sap_data`, `tsap_data`: added formal `Returns:` sections
- `components.py` `TransmissionSE.properties`: corrected return type annotation from `-> list` to `-> list[PropertyType]` (linter fix)

### Added (type annotations pass)
- Type annotations to all methods in `src/laser/cohorts/statearray.py`: `__new__` (`-> "StateArray"`, `dtype: Any`, `default_value: int | float`), `_cache_state_views` (`obj: "StateArray"`, `-> None`), `__array_finalize__` (`obj: np.ndarray | None`, `-> None`), `__getattr__` (`name: str`, `-> Any`), `__setattr__` (`name: str, value: Any`, `-> None`), `__getitem__` (`key: Any`, `-> np.ndarray`), `state_names` (`-> tuple[str, ...] | None`), `get_state_index` (`name: str`, `-> int | None`)
- `from typing import Any` import to `src/laser/cohorts/statearray.py`
- `model: Model` parameter type annotation to all 10 component `__init__` methods in `src/laser/cohorts/components.py` (via `TYPE_CHECKING` guard to avoid circular import)
- `from __future__ import annotations` and `TYPE_CHECKING` import guard to `src/laser/cohorts/components.py`
- `proposal: list` parameter type annotation to `Model.components` setter in `src/laser/cohorts/model.py`
- `-> None` return type annotations to all ~70 test methods in `tests/test_statearray.py`; fixture parameter types (`np.ndarray`) for all fixture-receiving test methods
- `-> None` return type annotations to `test_*` functions in all 8 model integration test files

### Fixed (type annotations pass)
- `components.py` `Exposed.states`: corrected return type annotation from `-> list` to `-> list[str]`

### Added (second pass — discovered in post-edit audit)
- Module docstrings to `src/laser/cohorts/SI.py`, `SIR.py`, `SIS.py`, `SIRS.py`, `SEI.py`, `SEIR.py`, `SEIS.py`, `SEIRS.py` (model preset files omitted from first pass)
- Module docstring to `src/laser/cohorts/vitaldynamics.py` (empty placeholder file)
- Module docstrings to all 8 model integration test files (`tests/test_sir.py` … `tests/test_seis.py`)

### Added (run_model params + CLI)
- Optional `params: dict | None` argument to all 8 `run_model()` helpers; defaults merged via `PropertySet({**defaults, **(params or {})})`
- Each `test_*` function now passes its parameter values explicitly via `params=` so tests remain valid if defaults change
- `if __name__ == "__main__"` blocks replaced with `argparse` CLIs accepting arbitrary `KEY=VALUE` positional arguments plus `--interactive`

### Fixed (stochastic test stability)
- Added `laser.core.random.seed()` to `test_seir`, `test_seirs`, `test_sir`, `test_sirs`, `test_seis`, and `test_sis` to prevent flaky failures caused by stochastic epidemic extinction in nodes with large populations relative to the 10-individual initial seed

### Added (NonDiseaseMortality)
- `src/laser/cohorts/vitaldynamics.py`: implemented `NonDiseaseMortality` component; accepts scalar, `ValuesMap`, or 2-D ndarray `mu`; applies binomial mortality draws to all states (or an optional named subset) each tick; registers `non_disease_mortality` node property
- `laser.cohorts.NonDiseaseMortality` exported from `__init__.py`
- `tests/test_mortality.py`: 8 new tests covering zero mortality, certain mortality, death accumulation, state-subset masking, default all-states behaviour, and all three `mu` input forms

### Changed (NonDiseaseMortality refactor)
- `components.py`: removed `mu` parameter and binomial mortality draws from `Susceptible`, `Exposed`, `Infectious`, `Recovered`, `InfectiousToRecovered`, `InfectiousToSusceptible`, and `RecoveredToSusceptible`; each component's `properties` no longer registers `non_disease_mortality`
- Updated stochastic test seeds (`test_sir` seed 2→0, `test_sis` seed 5→11) to account for the changed RNG call sequence after removing zero-rate binomial draws

### Changed (PropertyType consolidation)
- `src/laser/cohorts/utils.py`: added `PropertyType` type alias (was duplicated in `components.py` and `vitaldynamics.py`)
- `components.py` and `vitaldynamics.py`: removed local `PropertyType` definitions; now import from `laser.cohorts.utils`
- `laser.cohorts.PropertyType` exported from `__init__.py` for use in custom component authors

### Changed (component parameter renames)
- `components.py` `Exposed.__init__`: renamed parameter `sigma` → `r_progression`; updated `self.sigma` → `self.r_progression` and all internal references
- `components.py` `InfectiousToRecovered.__init__` and `InfectiousToSusceptible.__init__`: renamed parameter `gamma` → `r_recovery`; updated `self.gamma` → `self.r_recovery` and all internal references
- `components.py` `RecoveredToSusceptible.__init__`: renamed parameter `omega` → `r_waning`; updated `self.omega` → `self.r_waning` and all internal references
- All 7 affected model integration test files (`test_sir.py`, `test_sis.py`, `test_sirs.py`, `test_sei.py`, `test_seir.py`, `test_seis.py`, `test_seirs.py`): updated component constructor keyword arguments, local variable names, `PropertySet` keys, and docstrings to match new parameter names

### Changed (carry-forward unknown-state warning)
- `model.py` `Model.components` setter: emits `UserWarning` via `warnings.warn` for each name in `carry_forward_states` that is not found in the model's registered states (previously silently skipped)
- `tests/test_model.py` `test_carry_forward_unknown_state_name_emits_warning`: updated from "silently ignored" to assert one `UserWarning` is emitted containing the unknown name, and that valid states are still carried

### Changed (Model centralised carry-forward)
- `model.py` `Model.__init__`: added `carry_forward_states: Iterable[str] | None = None` parameter; stores as a `set` (or `None`); initialises `_carry_mask` to `slice(None)`
- `model.py` `Model.components` setter: builds `_carry_mask` (boolean ndarray or `slice(None)`) from `carry_forward_states` using `StateArray.get_state_index` after the StateArray is allocated; unknown state names are silently skipped
- `model.py` `Model.run`: carries forward `states[tick+1][mask] = states[tick][mask]` at the top of each tick before calling `start_step`
- `components.py` `Susceptible.start_step`, `Exposed.start_step`, `Infectious.start_step`, `Recovered.start_step`: replaced per-state carry-forward with no-ops; docstrings updated
- `tests/test_model.py`: 7 new tests covering default all-state carry-forward, multi-state carry-forward, selective list/tuple carry-forward, empty carry-forward, ordering (carry-forward before step), and unknown state name tolerance

### Added (ConstantPopBirths)
- `src/laser/cohorts/vitaldynamics.py`: implemented `ConstantPopBirths` component; reads `nodes.non_disease_mortality[tick]` and adds the per-node death count back into `states.S[tick+1]`; `properties` declares `non_disease_mortality`; `states` declares `["S"]`
- `laser.cohorts.ConstantPopBirths` exported from `__init__.py`
- `tests/test_births.py`: 7 tests covering death replenishment, population conservation over single and multiple states and ticks, zero births when no deaths, births landing only in S, standalone use without `NonDiseaseMortality`, and per-node death-to-birth accounting

### Changed (NonDiseaseMortality parameter rename)
- `vitaldynamics.py` `NonDiseaseMortality.__init__`: renamed parameter `mu` → `r_mortality`; updated `self.mu` → `self.r_mortality` and internal `step` reference
- `tests/test_mortality.py`: renamed all `mu=` keyword arguments to `r_mortality=`, local variables `mu`/`mu_array` → `r_mortality`, `ndm.mu` → `ndm.r_mortality`, test function names, and all docstring references

### Changed (NonDiseaseMortality vectorised step)
- `vitaldynamics.py` `NonDiseaseMortality.setup`: replaced per-state view caching (`_state_views`) with a single boolean mask array (`_state_mask`) built via `StateArray.get_state_index`
- `vitaldynamics.py` `NonDiseaseMortality.step`: replaced loop over individual state views with a single vectorised boolean-index selection, one `np.random.binomial` call across all active states, and `mortality.sum(axis=0)` to accumulate per-node totals

### Changed (NonDiseaseMortality states parameter)
- `vitaldynamics.py`: `NonDiseaseMortality.__init__` `states` parameter type widened from `set[str] | None` to `Iterable[str] | None` (accepts list, tuple, set, generator, or any iterable); iterable is materialized to a `set` in `__init__` to support one-shot iterables; `Optional` import replaced with `collections.abc.Iterable`
- `tests/test_mortality.py`: added `test_states_as_list_restricts_mortality` and `test_states_as_tuple_restricts_mortality`; updated existing set-form test docstring to note it tests set specifically

### Changed (initial infection seeding)
- All 8 model integration tests now seed infections as `max(min(25, pop), int(0.01 * pop))` per node instead of a fixed count of 10, making epidemic establishment reliable even in large-population nodes
- All 6 stochastic test seeds unified to 0 (was 0/0/0/0/0/11) after re-calibration against the new initial conditions
