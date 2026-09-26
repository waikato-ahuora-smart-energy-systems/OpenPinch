# Integration Test Instructions - Delivery Workflow

## Delivery orchestration

Run `tests/packaging/test_delivery_workflows.py` and
`tests/packaging/test_release_manifest.py` with seed `20260715`. Required
failures, cancellations, missing proof, wrong attempts, changed trees,
partial uploads, conflicting bytes, and invalid artifact archives must fail
closed. Recovery must retain the build identity while checking the latest
source-run validation jobs. The workflow lint check includes ShellCheck.

Local emulation does not prove GitHub scheduler behavior, cross-platform
installation, branch-protection activation, or OIDC publisher configuration.
Observe the real PR before activating the new required check. Publication
and remote settings require separate authorization.

## Purpose

Verify that targeting, utility placement, generated tutorials, CI lane
contracts, optional simulation engines, documentation, and packaged artifacts
continue to interact correctly after repeated work was removed.

## Scenario 1: Utility Placement and Notebook 19

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/application/test_utility_placement.py \
  tests/application/test_utility_placement_batch.py \
  tests/packaging/test_notebooks.py::test_utility_placement_has_one_executable_thermodynamic_notebook \
  tests/packaging/test_notebooks.py::test_utility_placement_notebook_uses_a_bounded_demonstration_search \
  tests/packaging/test_notebooks.py::test_base_profile_notebook_executes\[19_utility_placement_optimisation.ipynb\]
```

Expected result: the shared solved evidence remains copy-isolated, real process
and site optimization remains covered, and Notebook 19 produces feasible,
finite, zero-fallback solutions under the bounded search.

## Scenario 2: CoolProp and TESPy Service Boundaries

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/application/test_coolprop_hpr_audit.py \
  tests/e2e/test_hpr_mvr.py
uv run --no-sync pytest --hypothesis-seed=20260715 -q -m tespy
```

Expected result: 14 CoolProp audit cases and 55 HPR/MVR E2E cases pass; the
complete TESPy selection passes in its optional-engine environment.

## Scenario 3: CI and Marker Contracts

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/packaging/test_packaging_metadata.py \
  tests/packaging/test_reuse_develop_ci.py
```

Expected result: the ordinary selection excludes `solver`, `tespy`,
`performance`, and `docs`; each specialized category has exactly one workflow
owner and participates in the required PR/release gates.

## Scenario 4: Distribution Boundary

Build both distributions, install the wheel outside the checkout, then execute
`scripts/artifact_install_smoke.py` as described in `build-instructions.md`.
The installed package must expose Notebook 19 and must not resolve imports from
the source checkout.

## Cleanup

Test temporary directories are pytest-owned. Manually created wheel-smoke
environments under `/tmp` can be removed after verification. Build artifacts in
`dist/` are reproducible and ignored by Git.
