# Build Instructions

## Prerequisites

- Python 3.14.2 or newer and `uv`
- Dependencies synchronized from `uv.lock`
- Configured local solver executables for the complete solver-enabled gate

## Build package artifacts

```bash
uv build --out-dir /tmp/openpinch-critical-corrections-dist
```

Expected artifacts are `openpinch-0.6.0.tar.gz` and
`openpinch-0.6.0-py3-none-any.whl`.

## Build documentation

```bash
uv run sphinx-build -W --keep-going -b html docs /tmp/openpinch-critical-corrections-docs
```

The build must finish without warnings or errors.

## Static verification

```bash
uv run ruff check .
git diff --check
```

Both commands must finish without findings.
