# Build Instructions

## Prerequisites

- Python 3.14 and `uv`
- Repository development dependencies synchronized from `uv.lock`
- Local solver binaries for the complete solver-enabled suite

## Build and documentation

Run:

```bash
uv build --out-dir /private/tmp/openpinch-exact-replay-dist
uv run sphinx-build -E -a -W --keep-going -b html docs /private/tmp/openpinch-exact-replay-docs
```

The source archive and wheel must build without error and contain notebook 19.
Sphinx must complete with warnings treated as errors.

## Static verification

Run:

```bash
uv run ruff check .
git diff --check
```

Both commands must complete without findings.
