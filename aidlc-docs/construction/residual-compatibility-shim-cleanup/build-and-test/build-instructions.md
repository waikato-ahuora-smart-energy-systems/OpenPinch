# Build Instructions

## Prerequisites

- Python 3.14.
- uv with the repository lockfile synchronized.
- Hatchling and the development dependency group.
- Couenne and IPOPT on `PATH` only for the optional real-solver profile.
- No environment variables or external services are required for the non-solver, documentation, or package build gates.

## Build Steps

Install the locked development environment:

```bash
uv sync --group dev
```

Build a clean wheel and source distribution in a dedicated output directory:

```bash
uv run python scripts/build_dist.py --output-dir /private/tmp/openpinch-dist
```

Expected artifacts:

- `openpinch-0.5.2-py3-none-any.whl`
- `openpinch-0.5.2.tar.gz`

Build the documentation with all warnings treated as errors:

```bash
uv run sphinx-build -E -a -W --keep-going -b html docs /private/tmp/openpinch-docs
```

The successful build ends with `build succeeded.` and contains no `api-lib` page.

## Isolated Artifact Verification

Create a new virtual environment outside the checkout, install the wheel, and run:

```bash
python /path/to/OpenPinch/scripts/artifact_install_smoke.py --repo-root /path/to/OpenPinch
```

The smoke must report an import path under the isolated environment's `site-packages`, expose only `PinchProblem` and `PinchWorkspace` at the root, and find all packaged notebooks and sample cases.

## Troubleshooting

- A dependency resolution failure indicates an unsynchronized uv cache or lockfile; rerun `uv sync --group dev` with the repository's supported network/cache access.
- A solver-profile failure caused by a missing executable is not a package-build failure; install IDAES solver binaries before running `pytest -m solver`.
- A Sphinx warning is a blocking documentation failure because the build uses `-W`.
