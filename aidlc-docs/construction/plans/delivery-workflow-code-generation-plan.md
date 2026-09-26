# Delivery workflow code-generation plan

## Status and unit context

Part 1 is complete; Part 2 was approved with permission to commit and create
a PR to main after verification. This is the single source
of truth for implementation sequencing and checkbox progress.

Workspace: `/Users/timothyw/Github_Local/OpenPinch`. One brownfield delivery
unit implements D01-D09 and N01-N18. User stories, new application APIs,
frontend components, databases, and schema migrations are not applicable.
Functional, NFR, and infrastructure designs are approved. External dependencies
are GitHub Actions and existing TestPyPI/PyPI publishers. Publishing and remote
configuration changes are not authorized by implementation approval.

Existing local documentation changes belong to this workflow and must be
preserved. Inspect branch, worktree, and current upstream identities before
implementation; do not silently merge or reset to reconcile new remote work.
Do not spawn subagents without explicit authorization.

## Implementation steps

### Step 1 - Record executable baseline and inventory

- [x] Inspect current workflow/script/test owners and record baseline lane
  selectors, expanded matrices, coverage settings, version behavior, and
  targeted delivery-test results. Identify current hosted evidence separately
  from local runs. Read existing assertions before replacing them.

Files: `.github/workflows/ci-develop.yml`, `ci-pull-request.yml`,
`ci-publish.yml`; `tests/packaging/test_packaging_metadata.py`,
`test_reuse_develop_ci.py`, `test_package_index_release.py`,
`test_docs_consistency.py`; existing release scripts.

### Step 2 - Shared lane policy and gate contracts

- [x] Add `scripts/ci_policy.py` for versioned profiles, selectors, expected
  matrix cells, and pure gate decisions. Add
  `tests/packaging/test_ci_policy.py` with examples and generated invariants.
  Keep exact evaluated candidate/profile identity explicit in all interfaces.

Map: D03-D05, N03, N06-N10. Gate oracle tests reject missing, duplicate,
failed, cancelled, and unjustified skipped cells. Centralize reusable delivery
strategies in `tests/strategies/delivery.py` where multiple tests need them.

### Step 3 - Bounded package-index verification

- [x] Refactor `scripts/check_package_index_release.py` in place and extend
  `tests/packaging/test_package_index_release.py`. One retry owner, monotonic
  budget, configurable bounds, typed observations, bounded payloads, safe
  diagnostics, and injected transport/time must preserve exact hash checks.

Map: D06, N01-N02, N09, N14, N16. Preserve machine-readable status output on
stdout; send progress to stderr so shell preflight captures remain valid.
Test late responses, delays beyond 50 seconds, partial uploads, deadline
exhaustion, Retry-After, certificate/auth errors, malformed JSON, duplicates,
wrong hashes, and invalid timing parameters. No actual sleeping or network.

### Step 4 - Complete candidate proof and reuse

- [x] Update `scripts/reuse_develop_ci.py` to consume shared lane policy and
  verify candidate/profile/toolchain/provenance/run-attempt compatibility.
  Extend `tests/packaging/test_reuse_develop_ci.py` using fake API/git boundaries
  and generated evidence. Add bounded pagination and explicit fallback reasons.

Map: D04-D05, N03, N07-N10, N12, N15. Include performance evidence; reject
newer failed attempts, differing merge trees, incomplete matrices, and
truncated API evidence. Do not generalize cross-run reuse beyond proven cases.

### Step 5 - Release manifest and recovery decisions

- [x] Add `scripts/release_manifest.py` and
  `tests/packaging/test_release_manifest.py` for exact bundle identity,
  provenance validation, serialization, and pure recovery transitions.
  Reuse existing tag/version/build validation helpers where applicable.

Map: D07-D08, N04-N05, N12-N14, N17. Add round-trip, invariant, idempotence,
oracle, and stateful tests with realistic bounded inputs. Cover interruption
at each boundary, partial publication, expired artifacts, conflicting draft
assets, wrong source/run/attempt, and absent trusted manifest. No rebuilding
as recovery and no live external mutations in tests.

### Step 6 - Shared validation and branch callers

- [x] Create `.github/workflows/ci-validation.yml` and `ci-main.yml`; refactor
  `ci-develop.yml` to the shared owner. Preserve ordinary coverage and all
  docs, TESPy, performance, optional-install, artifact, and solver profiles.
  Add report/duration artifacts with bounded retention and stable lane IDs.

Map: D03, D09, N06-N10, N15, N17. Shared workflow inputs include explicit
candidate and profile, not mutable implicit head. Existing tools/actions and
runner matrix stay pinned/as configured. Publishing permissions are absent.

### Step 7 - Read-only PR workflow and complete merge gate

- [x] Refactor `.github/workflows/ci-pull-request.yml`: remove automatic
  version writes; implement candidate/base/metadata event planning, validation
  reuse, `OpenPinch PR Gate`, and transitional `test` compatibility check that
  depends on the full gate. Add `scripts/plan_ci_event.py` and focused tests in
  `tests/packaging/test_ci_events.py` if event logic needs a testable helper.

Map: D01-D05, N03, N07, N11-N13. Base edits validate the new merge candidate;
description edits do not start expensive lanes or cancel active validation.
No success without compatible evidence. Draft handling and cancelled/failed
dependencies are explicit. Do not change remote required checks here.

### Step 8 - Explicit release and same-artifact resume workflow

- [x] Refactor `.github/workflows/ci-publish.yml` to explicit main-context
  dispatch, validated immutable source/version, new-release versus resume
  inputs, full-profile evidence, exact artifact handoff, staging, both index
  verifications, and final GitHub publication. Remove main-push publication
  and unnecessary automatic self-dispatch permissions together.

Map: D01-D02, D06-D08, N01-N05, N11-N14, N17. Keep filename and `pypi`
environment; distinguish repository code readiness from pending publisher/ref
configuration verification. Serialize release mutations; no cancellation of
active publication. Recovery uses original manifest and bytes. Keep all write
and OIDC permissions job-scoped. No real dispatch or publication in this step.

### Step 9 - Workflow behavior contracts and lint

- [x] Add `tests/packaging/test_delivery_workflows.py`; update
  `test_packaging_metadata.py`, `test_reuse_develop_ci.py`, and
  `test_docs_consistency.py` assertions that encode the retired workflow.
  Test parsed structure plus helper behavior and mocked orchestration paths,
  not just text substrings. Add pinned actionlint validation in the shared
  quality lane after checking official installation/version documentation.

Map: all D requirements, especially D04/D08/D09. Verify triggers, scoped
permissions, explicit candidate references, required dependency paths,
artifact provenance, compatibility gate, timeout/retention configuration, and
finalization order. Retain substantive integrity tests when changing layout.
Use existing YAML tooling if available; if a dev dependency is needed, change
`pyproject.toml` and regenerate `uv.lock` without changing runtime dependencies.

### Step 10 - Developer and recovery documentation

- [x] Update `docs/developer/build-and-coverage.rst` and
  `docs/developer/index.rst`; add `docs/developer/releasing.rst` for version
  preparation through a normal PR, ordinary merges, explicit release/resume,
  retention limitations, and exact staged protection/publisher migration.
  Update documentation consistency tests for the new gate and policy.

Map: D09, N15-N18. Include a legacy 0.6.10 recovery runbook with prerequisites
and explicit authorization boundary; do not execute recovery. Store workflow
implementation summary under
`aidlc-docs/construction/delivery-workflow/code/implementation-summary.md`.

### Step 11 - Focused integration and design audit

- [x] Run all affected packaging/delivery examples and Hypothesis tests with
  deterministic seed and shrinking, Ruff, formatter checks, actionlint, YAML
  validation, and `git diff --check`. Reconcile every D01-D09/N01-N18 item with
  implementation and evidence. Audit untrusted-input and permission paths.

Failure tests must exercise real helper logic with fake boundaries. Do not
claim that local workflow emulation proves hosted scheduler/OIDC behavior.
Resolve discovered defects within the approved scope before full verification.

### Step 12 - Full verification and handoff evidence

- [x] Run ordinary branch coverage, specialized profiles supported by the
  local environment, documentation build, distribution build, and isolated
  artifact smoke. Compare lane collection against baseline for no lost or
  multiply-owned tests. Record unavailable solver/OS cases as hosted work,
  never as passed. Document timing without claiming unmeasured improvements.

Update the implementation summary, requirements traceability, code-generation
checkboxes, and workflow state. Keep hosted CI, required-check activation,
publisher configuration, and real release verification explicitly separate
from local implementation completion. Do not commit, push, merge, dispatch,
or modify remote settings without the corresponding user authorization.

## PBT compliance plan

- PBT-01: functional-design property catalogue mapped to Steps 2-5.
- PBT-02: manifest round trips in Step 5; document any lossy fields explicitly.
- PBT-03: gate, identity, file-map, and deadline invariants in Steps 2-5.
- PBT-04: recovery idempotence in Step 5.
- PBT-05: reference gate predicate and recovery model in Steps 2 and 5.
- PBT-06: generated recovery/interruption sequences in Step 5.
- PBT-07: reusable bounded domain strategies in Steps 2-5.
- PBT-08: shrinking retained, seed recorded, no flaky-test suppression; Step 11.
- PBT-09: existing declared Hypothesis/pytest stack retained.
- PBT-10: examples for the observed failure and every critical transition
  complement generated properties throughout Steps 2-5 and 9.

All PBT obligations have implementation evidence in the code summary.
Security and Resiliency extensions remain disabled. Explicit design security
and reliability constraints remain mandatory regardless.

## Planning completion

- [x] Map approved unit designs and requirements to existing repository owners.
- [x] Identify exact modified/new paths and dependency-ordered steps.
- [x] Include regression, property, workflow, and full-profile verification.
- [x] Specify documentation and remote activation boundaries.
- [x] Obtain explicit approval for the complete generation sequence.
