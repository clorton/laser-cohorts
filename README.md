# laser-cohorts
MPM/compartmental spatial, infectious disease models based on laser-core

## Development setup

- install `uv` in the system Python

```shell
python3 -m pip install uv
```

- create and activate a virtual environment for local development and excution
  - _you may choose any supported Python version >= 3.9_

```shell
uv venv --python 3.13 .venv
source .venv/bin/activate # MacOS and Linux
# .venv/bin/activate.bat # Windows
```

- install `tox` and to`x-uv` for running checks and tests

```shell
uv tool install tox --with tox-uv
```
