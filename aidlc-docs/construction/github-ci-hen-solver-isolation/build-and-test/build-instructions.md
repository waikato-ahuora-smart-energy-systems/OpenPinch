# Build Instructions: GitHub CI HEN Solver Isolation

## Prerequisites

- **Python**: 3.14.2, matching GitHub Actions.
- **Environment manager**: uv with the repository `.venv` and `uv.lock`.
- **Test dependencies**: The project development dependency group.
- **External HEN solvers**: Not required and must not be used for this focused
  non-solver verification.
- **Environment variables**: None.

## Environment Setup

From `/Users/timothyw/Github_Local/OpenPinch`:

```bash
/opt/homebrew/bin/uv sync --group dev
```

GitHub Actions uses the equivalent editable package and development-group pip
installation defined in `.github/workflows/ci-pull-request.yml`.

## Build Verification

This is a test-only Python patch and produces no new distribution artifact.
Successful pytest collection and import of the modified module provide the
relevant build verification:

```bash
/opt/homebrew/bin/uv run pytest --collect-only -q \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py
```

## Expected Result

- The module collects without syntax or import errors.
- No Couenne or IPOPT executable is consulted during the isolated test.
- No wheel, source distribution, dependency, or lockfile changes are expected.

## Troubleshooting

### uv Cache Permission Failure

Run the approved uv command in an environment that can access the configured uv
cache. Do not replace the locked environment or install unrelated dependencies.

### External Solver Error Reappears

Confirm the affected test accepts `monkeypatch` and invokes
`_use_fake_default_executor(monkeypatch)` before constructing the example
problem. Do not solve this by adding external solvers to the general CI job.
