# Package Architecture Modernization Build Instructions

## Prerequisites

- Python 3.14.2 or newer.
- `uv` with access to the repository lock file.
- No environment variable is required for the core build.
- Optional solver and presentation dependencies are needed only for their
  corresponding integration tests.

## Environment

From the repository root:

```bash
uv sync --group dev
```

## Static and Documentation Build

```bash
uv run ruff check .
uv run ruff format --check .
uv run python scripts/build_docs.py
```

The documentation command must complete with no Sphinx warning.

## Isolated Distribution Build

```bash
uv run python -m build --wheel --sdist --outdir /private/tmp/openpinch-dist
```

Expected artifacts:

- `openpinch-0.5.0-py3-none-any.whl`
- `openpinch-0.5.0.tar.gz`

Inspect the wheel with `unzip -Z1` and the source distribution with `tar -tzf`.
Both must include `main.py` and all seven owner packages and must omit the
retired `classes`, `lib`, `services`, `utils`, and `streamlit_webviewer` trees.

## Clean-Install Verification

Create a fresh environment outside the checkout, install the wheel plus pytest
and Hypothesis, and run `scripts/artifact_install_smoke.py`. Copy
`tests/e2e/test_main.py`, `tests/support`, and `examples/stream_data` outside the
checkout before running the 59-case external suite with `-W error`.

## Troubleshooting

- If `uv` cannot access its cache, grant the build process access to the user
  cache or select a writable `UV_CACHE_DIR`.
- If a solver test skips, inspect its reported optional dependency or benchmark
  exclusion. Do not count an unavailable solver as a pass.
- If an artifact resolves OpenPinch from the checkout, rerun from a temporary
  directory with the clean environment's Python executable.
