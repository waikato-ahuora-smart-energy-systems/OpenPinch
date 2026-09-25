# HPR and MVR Notebook Reliability Build Instructions

## Prerequisites

- CPython 3.14.2 or a later compatible 3.14 release.
- `uv` with access to the committed `uv.lock`; verification used 0.11.29.
- The locked `dev` dependency group and Hatchling build backend.
- No external optimisation solver is required for this workflow.
- CoolProp is required by the real vapour-compression notebook profile.

Run commands from the repository root unless a step states otherwise.

## Build Steps

### 1. Synchronize the locked development environment

```bash
uv sync --frozen --group dev
uv run --no-sync python scripts/check_lockfile_version.py
```

### 2. Validate source and generated artifacts

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check \
  scripts/generate_tutorial_notebooks.py \
  scripts/generate_tutorial_coverage.py \
  tests/packaging/test_notebooks.py \
  tests/packaging/test_hpr_notebook_properties.py \
  tests/packaging/test_docs_consistency.py \
  tests/packaging/test_tutorial_coverage.py
uv run --no-sync python -m compileall -q OpenPinch scripts tests
git diff --check
```

### 3. Build warning-strict documentation

```bash
uv run --no-sync python scripts/build_docs.py
```

The command treats Sphinx warnings as errors and writes the RTD-compatible HTML
tree to `docs/_build/html`.

### 4. Build the wheel and source distribution

```bash
uv run --no-sync python scripts/build_dist.py
```

For release 0.6.9 the expected artifacts are:

- `dist/openpinch-0.6.9-py3-none-any.whl`
- `dist/openpinch-0.6.9.tar.gz`

### 5. Verify the installed wheel outside the checkout

Create an unused temporary environment path, install the built wheel with its
declared dependencies, change outside the repository, and run:

```bash
python /absolute/path/to/OpenPinch/scripts/artifact_install_smoke.py --surface core
```

The reported `OpenPinch.__file__` must be inside the temporary environment's
`site-packages`, not the source checkout.

## Successful Build Evidence

- Lockfile and project version: synchronized at 0.6.9.
- Ruff lint, changed-file formatting, compilation, and patch hygiene: passed.
- Warning-strict Sphinx 9.1 build: passed for 57 RST sources.
- Wheel and source distribution: built successfully.
- All four HPR/MVR tutorial notebooks: present in the wheel.
- Isolated core-wheel runtime smoke: passed on CPython 3.14.2.

## Troubleshooting

### `uv` is unavailable outside the checkout

Use the absolute path returned by `command -v uv`. A temporary working
directory may not inherit the interactive shell's package-manager path.

### Notebook execution reports a missing optional dependency

Synchronize the declared development environment. The four required HPR/MVR
notebooks need CoolProp; TESPy and Brayton demonstrations remain optional and
publish typed status evidence when their dependency or method is unavailable.

### A generated notebook differs from its source owner

Run `uv run --no-sync python scripts/generate_tutorial_notebooks.py`, review
the resulting notebook diff, and rerun the generator/static tests. Do not edit
the generated notebook JSON as an independent source.
