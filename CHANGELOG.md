# Changelog

## Unreleased

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

### Changed (NonDiseaseMortality states parameter)
- `vitaldynamics.py`: `NonDiseaseMortality.__init__` `states` parameter type widened from `set[str] | None` to `Iterable[str] | None` (accepts list, tuple, set, generator, or any iterable); iterable is materialized to a `set` in `__init__` to support one-shot iterables; `Optional` import replaced with `collections.abc.Iterable`
- `tests/test_mortality.py`: added `test_states_as_list_restricts_mortality` and `test_states_as_tuple_restricts_mortality`; updated existing set-form test docstring to note it tests set specifically

### Changed (initial infection seeding)
- All 8 model integration tests now seed infections as `max(min(25, pop), int(0.01 * pop))` per node instead of a fixed count of 10, making epidemic establishment reliable even in large-population nodes
- All 6 stochastic test seeds unified to 0 (was 0/0/0/0/0/11) after re-calibration against the new initial conditions
