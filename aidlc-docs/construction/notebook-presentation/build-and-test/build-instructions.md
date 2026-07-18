# Notebook Presentation Build Instructions

## Prerequisites

- Python 3.14.2 or newer.
- uv with the committed `uv.lock` environment.
- Development, notebook, plotting, dashboard, Brayton, synthesis, and build
  dependencies from `pyproject.toml` for the complete profile matrix.
- Provisioned external HEN solver binaries for the solver tutorial profile.
- No database, service, container, or network runtime is required once the
  locked dependencies are available.

## Build Steps

Install or refresh the locked environment when required:

```bash
uv sync --all-extras --dev
```

Regenerate the canonical notebook resources:

```bash
uv run python scripts/generate_tutorial_notebooks.py
```

Build documentation in a clean destination:

```bash
uv run python scripts/build_docs.py --output-dir /path/to/clean/docs-html
```

Build the wheel and source archive in a clean destination:

```bash
uv run python scripts/build_dist.py --output-dir /path/to/clean/dist
```

## Success Criteria

- Canonical generation produces exactly 18 byte-stable source-only notebooks.
- Sphinx completes with zero warnings and writes `index.html`.
- Distribution output contains one OpenPinch 0.5.3 wheel and one source archive.
- The wheel contains all 18 improved tutorial resources.

## Troubleshooting

- Run uv commands with access to the existing user cache if restricted cache
  permissions prevent environment startup.
- Treat every Sphinx warning as a build failure.
- Install the declared optional group before interpreting a profile import as a
  notebook defect.
- Verify the external solver separately from Python dependency installation.
