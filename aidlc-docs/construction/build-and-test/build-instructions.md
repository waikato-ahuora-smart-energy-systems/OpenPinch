# Build Instructions

## Prerequisites

- **Build tool**: `uv`, Python build frontend, and Hatchling.
- **Python**: CPython 3.14.2 or later, as declared by `pyproject.toml`.
- **Core dependencies**: NumPy, Pint, pandas, CoolProp, Pydantic, and SciPy.
- **Development dependencies**: pytest, Hypothesis, coverage, Ruff, Sphinx,
  `build`, and Hatchling from the locked development group.
- **Optional dependencies**: notebook extras for an interactive Jupyter
  session; solver extras only for solver-marked HEN tests.
- **Environment variables**: none for the utility-placement core, build, or
  base-profile notebook. External solver executable variables are required
  only for the separately marked solver suite.
- **System requirements**: any supported operating system with a Python 3.14
  environment; temporary disk space for one source archive and wheel.

## Build Steps

### 1. Install Dependencies

From the repository root:

```bash
uv sync --all-extras --group dev
```

For the core feature and base notebook only, the locked default and development
dependencies are sufficient; solver executables are not needed.

### 2. Configure Environment

No runtime configuration is needed for utility placement. Confirm the selected
interpreter and import source:

```bash
uv run python --version
uv run python -c "import OpenPinch; print(OpenPinch.__file__)"
```

### 3. Build All Units

Use installed locked build tooling without dependency download:

```bash
uv run python -m build --no-isolation --outdir dist
```

### 4. Verify Build Success

- **Expected output**: `Successfully built openpinch-0.5.4.tar.gz and
  openpinch-0.5.4-py3-none-any.whl` for the current project version.
- **Build artifacts**: `dist/openpinch-*.tar.gz` and
  `dist/openpinch-*-py3-none-any.whl`.
- **Required archive content**: `OpenPinch/contracts/utility_placement.py`, all
  `OpenPinch/analysis/utility_placement/` modules,
  `OpenPinch/application/utility_placement.py`,
  `OpenPinch/application/_problem/accessors/target.py`, and
  `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`.
- **Common warnings**: none are accepted from the build itself. Shell startup
  warnings about a restricted process list are environment noise, not build
  output.

## Troubleshooting

### Build Fails with Dependency Errors

- **Cause**: isolated build mode tries to download Hatchling in a network-
  restricted environment, or the lock is not synchronized.
- **Solution**: run `uv sync --all-extras --group dev`, then use
  `python -m build --no-isolation` through the synchronized environment.

### Build Fails with Compilation or Package-Data Errors

- **Cause**: invalid Python syntax, stale generated notebook output, or missing
  Hatchling package-data discovery.
- **Solution**: run Ruff and the focused packaging/notebook tests, regenerate
  notebooks with `uv run python scripts/generate_tutorial_notebooks.py`, and
  rebuild. Do not hand-edit the generated notebook.
