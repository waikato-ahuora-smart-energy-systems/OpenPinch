# Build Instructions

## Prerequisites

- Python 3.11 or newer.
- `uv` with the repository lock file available.
- Network access only when locked dependencies are not already installed.
- Chrome or Chromium plus Kaleido for static Plotly image tests.
- External HEN solvers only when running the separately marked solver profile.

No application environment variables or external services are required for the
base package build.

## Build Steps

Install the locked development environment:

```bash
uv sync --all-extras --dev
```

Build the wheel and source distribution:

```bash
uv run python scripts/build_dist.py --output-dir dist
```

Build warning-free HTML documentation:

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Successful distribution output contains one `openpinch-*.whl` and one
`openpinch-*.tar.gz`. Successful documentation output is under
`docs/_build/html` with zero warnings.

## Troubleshooting

- The Sphinx tree is self-contained and must build without network access.
- Kaleido browser failures indicate that Chrome cannot start in the current
  sandbox; verify image export in an environment allowed to launch it.
- HEN solver failures belong to the explicit `solver` test profile and require
  the corresponding solver installation.
