# Build Instructions - Delivery Workflow

## Automatic version preparation

Use Python 3.14.2 and the existing locked dev environment. Check metadata with
`python scripts/check_release_version.py` and
`python scripts/check_lockfile_version.py`. Build with
`python scripts/build_dist.py --output-dir <new-empty-temporary-directory>`;
do not overwrite recovery distributions or upload this local verification build.
Run Ruff on changed Python files and actionlint with ShellCheck on workflows.
No application dependency or project version bump is part of this implementation.

## Delivery-specific validation

For automatic-release restoration, application code and dependencies are
unchanged. Reproduce local verification with the commands below; no release
dispatch or local artifact upload is part of this test procedure.

```bash
uv run --no-sync ruff check scripts/plan_release.py scripts/release_manifest.py tests/packaging/test_plan_release.py tests/packaging/test_delivery_workflows.py tests/packaging/test_docs_consistency.py
uv run --no-sync ruff format --check scripts/plan_release.py scripts/release_manifest.py tests/packaging/test_plan_release.py tests/packaging/test_delivery_workflows.py tests/packaging/test_docs_consistency.py
actionlint
git diff --check
```

Use actionlint 1.7.12 with ShellCheck installed. The automatic publisher is a
standalone completion listener, not a reusable trusted-publisher workflow.
Keep `ci-publish.yml` and the protected `pypi` environment unchanged in publisher
configuration. A successful local build is not authorization to publish.

The shared workflow is `.github/workflows/ci-validation.yml`; callers select
integration or full profiles from `scripts/ci_policy.py`. Run pinned
actionlint 1.7.12 with ShellCheck over all workflows, then the existing static
and build commands below. Artifact smoke environments must be separate for
core and TESPy; the core check intentionally rejects installed TESPy.

Do not dispatch the publishing workflow as a test. Local builds are not
trusted release bundles: CI supplies immutable source/run/attempt identity.
Release initiation, resume, and configuration activation are documented in
`docs/developer/releasing.rst`.

## Prerequisites

- Python 3.14.2, matching `pyproject.toml` and CI.
- uv 0.11.29, matching the GitHub Actions workflows.
- Hatchling and `build` from the locked development dependency group.
- macOS, Linux, or Windows for the core wheel smoke. CI uses Ubuntu for the
  TESPy, performance, documentation, and solver lanes.
- Temporary disk space for `.venv/`, `dist/`, documentation output, coverage
  data, and isolated installed-wheel environments.

The core build requires no credentials or network service. The solver lane
additionally requires usable Couenne and IPOPT binaries. Local installations
may expose solver paths through `IPOPT_EXECUTABLE`, `CBC_EXECUTABLE`,
`BONMIN_EXECUTABLE`, `COUENNE_EXECUTABLE`, `APOPT_EXECUTABLE`,
`MOJOPSE_EXECUTABLE`, and `AMPLFUNC` when those tools are not discoverable on
`PATH`.

## Build Steps

### 1. Synchronize the Locked Environment

```bash
uv sync --frozen --group dev
uv run --no-sync python scripts/check_lockfile_version.py
```

Expected result: the Python 3.14.2 environment is synchronized without
changing `uv.lock`, and the package version agrees across project metadata.

### 2. Run Static Validation

```bash
uv run --no-sync ruff check .
git diff --check
```

Expected result: both commands exit zero.

### 3. Build the Wheel and Source Distribution

```bash
uv run --no-sync python scripts/build_dist.py
```

Expected artifacts for version 0.6.10:

- `dist/openpinch-0.6.10-py3-none-any.whl`
- `dist/openpinch-0.6.10.tar.gz`

Both archives must contain
`OpenPinch/tutorials/notebooks/19_utility_placement_optimisation.ipynb`.

### 4. Verify an Installed Artifact

Use a clean virtual environment outside the repository so an editable checkout
cannot satisfy the import accidentally:

```bash
uv venv --python 3.14.2 /tmp/openpinch-wheel-smoke/.venv
uv pip install --python /tmp/openpinch-wheel-smoke/.venv/bin/python \
  dist/openpinch-0.6.10-py3-none-any.whl
cd /tmp/openpinch-wheel-smoke
.venv/bin/python /path/to/OpenPinch/scripts/artifact_install_smoke.py \
  --repo-root /path/to/OpenPinch --surface core
```

Expected result: the smoke reports an import path inside the isolated virtual
environment and validates the CLI, packaged resources, root API, and targeting
workflow.

## Troubleshooting

### Dependency or Lock Failure

- Confirm Python is exactly 3.14.2 and uv is compatible with 0.11.29.
- Run `uv lock --check`; do not regenerate the lock merely to bypass a mismatch.
- Install the required optional extra when running a specialized surface.

### Artifact Smoke Imports the Checkout

- Run the smoke from outside the repository.
- Recreate the temporary virtual environment and reinstall the wheel.
- Do not set `PYTHONPATH` to the checkout.

### Solver Selection Skips or Fails

- Run `uv run --no-sync idaes get-extensions` on a supported platform.
- Verify Couenne and IPOPT with `pyomo.environ.SolverFactory` before pytest.
- Treat missing external binaries as environment setup, not an ordinary-lane
  failure.
