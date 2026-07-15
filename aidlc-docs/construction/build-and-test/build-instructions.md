# Build Instructions

## Environment

Use Python 3.14.2 or newer and the locked development environment from the repository root.

```bash
uv sync --group dev
```

## Distribution Build

```bash
uv run python scripts/build_dist.py
```

The verified acceptance build produced both the OpenPinch 0.4.5 wheel and source distribution. A temporary output directory may be selected with `--output-dir` to avoid replacing local release artifacts.

## Documentation Build

```bash
uv run sphinx-build -W -b html docs docs/_build/html
```

The segmented-stream API and input guide build warning-free.
