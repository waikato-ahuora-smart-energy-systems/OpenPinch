# Build Instructions

## Prerequisites

- Python version selected by `.python-version`.
- uv with the committed `uv.lock` environment available.
- Hatchling and `build` from the development dependency group.
- Sphinx and the Read the Docs theme from the documentation/development groups.
- No application environment variables, database, service, or network access
  when the locked environment and package caches are already available.

External HEN solver binaries are required only for separately marked solver
profiles and are not needed for this remediation build.

## Build Steps

Install or refresh the locked environment when needed:

```bash
uv sync --all-extras --dev
```

Create clean temporary destinations rather than reusing ignored build trees:

```bash
mktemp -d /private/tmp/openpinch-docs.XXXXXX
mktemp -d /private/tmp/openpinch-dist.XXXXXX
```

Build warning-as-error HTML documentation:

```bash
uv run python scripts/build_docs.py --output-dir /path/to/clean/docs-html
```

Build the wheel and source archive:

```bash
uv run python scripts/build_dist.py --output-dir /path/to/clean/dist
```

## Success Criteria

- Sphinx completes with zero warnings and writes an HTML index.
- Distribution output contains exactly one `openpinch-*.whl` and one
  `openpinch-*.tar.gz` for the current version.
- Build logs contain no missing-file, import, or package-content failures.

## Troubleshooting

- A uv cache permission error requires running the approved uv command with
  access to the existing user cache; do not redirect or recreate the cache in
  the repository.
- A Sphinx warning is a build failure; correct the source/reference and rebuild
  from a fresh destination.
- A missing build backend means the locked development group was not installed.
- Optional solver errors belong only to explicitly selected solver profiles.
