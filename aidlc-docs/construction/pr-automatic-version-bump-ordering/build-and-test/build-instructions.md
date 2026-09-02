# Build Instructions

## Prerequisites

- Python 3.14.2 and uv 0.11.29.
- The repository's frozen development environment.
- Ruby with Psych for an independent local YAML syntax check.

## Build Steps

```bash
uv sync --frozen --group dev
uv run --no-sync python scripts/build_docs.py
```

The documentation build must complete without warnings. This workflow-only
correction produces no package artifact and intentionally does not change the
project version locally.

## Troubleshooting

- If uv reports a lock mismatch, confirm that no local version file changed.
- If Sphinx fails, resolve the reported reStructuredText warning before merging.
- If GitHub cannot push the bump commit, verify that Actions has repository
  write permission and that branch protection permits the GitHub Actions bot.
