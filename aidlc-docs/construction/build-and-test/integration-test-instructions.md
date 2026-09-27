# Integration Test Instructions - Delivery Workflow

## Version-preparation integration

The packaging suite uses temporary real Git objects and fake GitHub boundaries
to exercise create, duplicate reconcile, partial PR-create recovery, closed PR
and unrelated-source rejection. Review proof tests verify job/run/artifact/merge
identity and publisher behavior for successful reuse versus failed required jobs.
No remote source or package index is mutated by these tests.

Hosted activation is separate: verify Actions PR creation policy, current review
decision availability, bot PR-run approval, required complete gate, normal
two-merge path and trusted-publisher configuration. Install workflows together;
record the one-time bootstrap boundary and do not weaken branch protections.

## Delivery orchestration

### Automatic-release acceptance scenarios

- A successful same-repository main push uses its original validated bundle.
- Failed/cancelled/unrelated events do not publish; invalid successful-run
  evidence fails closed rather than being interpreted as absence.
- A later same-version commit verifies the original release bytes before no-op.
- Drafts, prereleases, partial indexes, expired/ambiguous evidence, tag-only
  state and conflicting bytes stop with verified-recovery guidance.
- Manual new/resume paths retain their original build and verification rules.
- The bundle's actual boolean condition is exercised across 72 combinations
  of mode, decision, request result and validation result.

Run these scenarios with fake GitHub/index boundaries through
`test_plan_release.py`, `test_release_manifest.py` and
`test_delivery_workflows.py`, using seed `20260926`. No remote writes occur.

### Hosted activation handoff (not performed locally)

After an authorized commit/PR and reviewed merge, verify that `Release and
Publish` starts from successful `CI Main` completion on the default branch.
Check that the reported source run, commit and artifact match the validated
candidate, and that TestPyPI verification precedes PyPI and GitHub finalization.
Inspect trusted-publisher/environment compatibility without weakening existing
protections. A real upload is irreversible and is not a diagnostic test.

The committed version is still 0.6.10. Its existing legacy tag/release is not
automatically adopted or repaired. Prepare a reviewed new version consistently
in project metadata, bump configuration and lockfile, or separately authorize
verified legacy recovery. Do not overwrite a distribution or move a tag.

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
