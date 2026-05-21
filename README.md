# laser-cohorts
MPM/compartmental spatial, infectious disease models based on laser-core

## Development setup

- install `uv` in the system Python directly with `pip` (below) or as Astral [recommends](https://github.com/astral-sh/uv#installation)

```shell
python3 -m pip install uv
```

- create and activate a virtual environment for local development and excution
  - _you may choose any supported Python version >= 3.10_

```shell
uv venv --python 3.13 .venv
source .venv/bin/activate # MacOS and Linux
# .venv/bin/activate.bat # Windows
```

- install `laser-cohorts` locally in development mode

```bash
uv pip install -e ".[dev]"
```

- install `tox` and `tox-uv` for running checks and tests

```shell
uv tool install tox --with tox-uv
```

_or_ run `tox` directly with `uvx`:

```bash
uvx --with tox-uv tox
```

## Documentation

Initial documentation [here](https://clorton.github.io/laser-cohorts/)
