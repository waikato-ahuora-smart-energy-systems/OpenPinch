# HPR and MVR Benchmark E2E Build Instructions

## Prerequisites

- **Python**: CPython 3.14.2 or newer within the supported 3.14 series.
- **Environment manager**: `uv`; verification used version 0.11.29.
- **Build backend**: Hatchling 1.26 or newer.
- **Runtime dependencies**: NumPy, pandas, Pint, CoolProp 8 or newer,
  Pydantic, and SciPy, as declared in `pyproject.toml`.
- **Development dependencies**: the locked `dev` dependency group.
- **Environment variables**: none are required for the HPR/MVR benchmark gate.
- **External solvers**: not required. The benchmark belongs to the normal
  `not solver` tier.
- **System requirements**: enough memory and disk for the locked Python
  environment, documentation output, coverage data, wheel, and source
  distribution. The project does not declare a fixed minimum.

Run all commands from the repository root.

## Build Steps

### 1. Install the locked development environment

```bash
uv sync --frozen --group dev
```

### 2. Run static checks and compilation

```bash
uv run --no-sync ruff check .
uv run --no-sync python -m compileall -q OpenPinch tests/e2e
```

For the files changed by this unit, also verify formatting without rewriting:

```bash
uv run --no-sync ruff format --check \
  OpenPinch/analysis/heat_pumps/service.py \
  OpenPinch/analysis/heat_pumps/common/postprocessing.py \
  tests/analysis/heat_pumps/test_targeting.py \
  tests/analysis/heat_pumps/test_hpr_residual_regressions.py \
  tests/e2e/cases.py \
  tests/e2e/hpr_benchmark.py \
  tests/e2e/test_hpr_benchmark_helpers.py \
  tests/e2e/test_hpr_mvr.py \
  tests/e2e/test_main.py
```

### 3. Build warning-strict documentation

```bash
uv run --no-sync python scripts/build_docs.py
```

Expected output is `docs/_build/html` with no Sphinx warnings.

### 4. Build distributions

```bash
uv run --no-sync python scripts/build_dist.py
```

Expected artifacts for version 0.6.9 are:

- `dist/openpinch-0.6.9-py3-none-any.whl`
- `dist/openpinch-0.6.9.tar.gz`

### 5. Verify the built wheel outside the source checkout

Create a temporary Python 3.14 environment, install the wheel with its declared
dependencies, and run these checks from outside the repository:

```bash
python -I -c 'import OpenPinch; from OpenPinch import PinchProblem; p = PinchProblem("process_mvr.json", project_name="Wheel Smoke"); assert type(p).__name__ == "PinchProblem"'
openpinch --help
```

The import path must resolve to the temporary environment's `site-packages`,
not the source checkout.

## Successful Build Evidence

- Ruff lint: passed.
- Changed-file formatting: nine files passed.
- Python compilation: passed.
- Warning-strict Sphinx build: passed, 57 sources processed.
- Wheel and source distribution: built successfully.
- Isolated wheel import, packaged `process_mvr.json` construction, and CLI help:
  passed.

## Troubleshooting

### `uv` is unavailable in a temporary working directory

Use the absolute path returned by `which uv`, or activate the shell environment
that provides `uv` before creating the temporary environment.

### The isolated import reports a missing runtime package

Install the wheel normally so its declared dependencies are resolved. A
`--no-deps` installation is suitable only for archive inspection and is not a
valid runtime smoke environment.

### The build uses the wrong Python version

Confirm `uv run --no-sync python --version` reports Python 3.14.2 or newer. The
system `python3` may be older than the project environment.

### CoolProp cannot evaluate a fluid state

Confirm CoolProp 8 or newer is installed from the locked environment. A
physical candidate failure in the benchmark must be returned as a bounded
`HPRTargetingError`; an import or dependency failure is an environment error.
