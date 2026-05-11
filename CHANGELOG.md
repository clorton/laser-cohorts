# Changelog

## Unreleased

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

### Added (second pass — discovered in post-edit audit)
- Module docstrings to `src/laser/cohorts/SI.py`, `SIR.py`, `SIS.py`, `SIRS.py`, `SEI.py`, `SEIR.py`, `SEIS.py`, `SEIRS.py` (model preset files omitted from first pass)
- Module docstring to `src/laser/cohorts/vitaldynamics.py` (empty placeholder file)
- Module docstrings to all 8 model integration test files (`tests/test_sir.py` … `tests/test_seis.py`)
