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
