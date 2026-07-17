# Integration Test Instructions

## Process-Engineer Workflow Scenarios

Run the architecture, package, documentation, notebook, and root workflow
integration checks:

```bash
uv run pytest -q tests/architecture tests/packaging tests/e2e/test_main.py
```

These tests verify:

- package-root `PinchProblem` and `PinchWorkspace` imports;
- explicit target, design, component, plot, report, and workspace interactions;
- exact operation-manifest parity with all eighteen tutorials;
- `HeatExchangerNetwork.model_dump(mode="json")` transport through
  `TargetInput.network`;
- workspace schema version 3 case-input persistence;
- warning-free RTD source consistency and packaged resource inventory;
- wheel and source-distribution package boundaries.

## Installed-Artifact Scenario

Build a wheel, install it with declared dependencies into a clean temporary
environment, then run:

```bash
python scripts/artifact_install_smoke.py --repo-root /path/to/OpenPinch
```

Run the script outside the checkout. It must import from site-packages, solve a
direct target, construct a workspace, expose exactly the two root workflow
classes, find all packaged assets, and execute CLI help.

Temporary build environments can be removed after verification; no service or
database cleanup is required.
