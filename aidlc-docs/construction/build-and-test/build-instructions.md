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

## GitHub CI Heat-Pump Zero-Duty Follow-Up

### Verified Environment

- Python 3.14.2
- NumPy 2.4.6
- CoolProp 7.2.0
- pytest 9.1.1
- Hypothesis 6.156.6

### Isolated Distribution Build

```bash
uv run python scripts/build_dist.py \
  --output-dir /private/tmp/openpinch-ci-build-20260715T2024
```

Verified result: OpenPinch 0.4.5 wheel and source distribution built
successfully without replacing workspace release artifacts.
