# Delivery implementation evidence

## Baseline

- Source: main 2e743566; dedicated branch codex/delivery-workflow-overhaul.
- Existing delivery selection: 60 tests passed in 6.17 seconds.
- Ordinary marker: not solver and not tespy and not performance and not docs;
  fixed Hypothesis seed 20260715; 95-percent branch coverage requirement.
- Separate docs, TESPy, performance, and solver lanes. Optional surfaces:
  core, dashboard, notebook, brayton_cycle, tespy, synthesis. Artifact smoke:
  Ubuntu, Windows, macOS plus TESPy. Preserve every selection.
- User's local Notebook 10 edit is unrelated and excluded from staging.

## Implementation and traceability

| Requirements | Implementation and executable evidence |
|---|---|
| D01-D02, N11-N13 | Read-only PR/develop/main callers; explicit main-context release dispatch; version preparation remains a reviewed source change. Parsed event/permission contracts and executable PR gate. |
| D03-D05, N03, N06-N10 | Shared lane policy and reusable validator; all six optional surfaces and three artifact OS cells retained; exact-tree develop reuse includes performance, policy, and attempt checks. Generated gate oracle, missing/duplicate/failure cases, stale attempts, pagination bounds. |
| D06, N01-N02, N14, N16 | One monotonic 300-second default postflight budget, 60-second preflight, bounded requests/payloads, permanent-versus-transient classification, safe destination/progress diagnostics. Fake-clock tests include 290-second visibility, late success, Retry-After exhaustion, TLS/auth failures, duplicate/conflicting files. |
| D07, N04-N05, N12-N14, N17 | Immutable manifest ties repository, source/tree, original build run/attempt, exact file hashes, artifact ID/digest. Actual downloaded archive digest and entries are verified. Serialized publishing, exact annotated tags, append-only matching draft assets, no clobber. Generated archive round trips, malicious paths, expired artifacts, real staging with fake commands and interruption at each asset boundary. |
| D08, N04-N05 | Source proof uses latest validation jobs so failed-job retries can retain the original build. Both indexes must verify before GitHub finalization. Already-public exact releases remain no-ops, including latest-release metadata. CLI failure-matrix tests and generated recovery model. |
| D09, N15-N18 | JUnit/duration artifacts retained 30 days, candidate/policy/reuse summaries, recovery identity, explicit release/resume and staged protection/publisher runbooks. Docs consistency and warning-strict Sphinx checks. |

## Verification

- Complete ordinary coverage run: 3451 passed, 6 expected skips, 65 specialized
  deselections in 382.68 seconds; 95-percent branch-aware coverage gate passed.
  Final expanded release-finalization tests were subsequently included in the
  packaging run below. Initial sandboxed run failed Chrome launch and a test
  collected before its filesystem timing allowance correction; both resolved.
- Corrected image export and artifact tests: 19 passed outside the sandbox.
- Final complete packaging selection: 273 passed, 6 expected skips in 63.17
  seconds, including the warning-strict Sphinx build and finalization matrix.
- Final marker partition: 3485 ordinary, 58 TESPy, 2 performance, 1 docs, and
  4 solver cases; zero overlapping specialized assignments. The ordinary count
  includes six expected skips and 28 finalization cases added after the full
  coverage run, all covered by the final packaging run.
- Specialized selection: 64 passed, 1 existing solver skip, 3451 deselected in
  214.30 seconds. Includes 58 TESPy, 2 performance, 1 docs, and 4 solver cases.
- Wheel and sdist built successfully. Separate external core and TESPy wheel
  environments passed installed-artifact checks; neither imported the checkout.
- Ruff, changed-file formatting, lock consistency, actionlint 1.7.12 with
  ShellCheck 0.11.0, and patch hygiene pass. Ten unrelated existing formatting
  differences are left untouched. No runtime dependency/version changed;
  PyYAML 6.0.3 is a new development-only dependency for parsed workflow tests.
- No numerical assertions, optimization budgets, benchmark cases, or marker
  assignments were weakened. No comparable hosted runtime improvement is claimed.

## PBT compliance

| Rule | Status | Evidence |
|---|---|---|
| PBT-01 | Compliant | Approved functional property catalogue mapped above. |
| PBT-02 | Compliant | Generated manifest JSON and actual archive round trips. |
| PBT-03 | Compliant | Gate, provenance, digest, deadline, and no-premature-finalization invariants. |
| PBT-04 | Compliant | Repeated real staging with fake external state preserves bytes; repeated completed recovery decisions stable. |
| PBT-05 | Compliant | Independent gate predicate and recovery decision oracle. |
| PBT-06 | Compliant | Generated interruption/resume state-machine sequences plus real staging interrupted at generated upload boundaries. |
| PBT-07 | Compliant | Reusable bounded release manifests, versions, SHAs, and constrained evidence/state domains. |
| PBT-08 | Compliant | Seed 20260715, shrinking retained, fake-time polling; filesystem integration allowance bounded at 2 seconds/example. |
| PBT-09 | Compliant | Existing pytest/Hypothesis dependency stack retained. |
| PBT-10 | Compliant | Explicit delayed visibility, auth/TLS, archive paths, retry attempts, finalization failures, and workflow contracts complement generated tests. |

Security and Resiliency extensions are disabled (N/A). Explicit approved
security and reliability requirements are nevertheless implemented and tested.

## Remaining operational boundaries

- No package publication, legacy 0.6.10 recovery, merge, or remote configuration
  change has been performed. Hosted scheduler/OS/OIDC behavior is not proven
  by local tests. Observe the PR checks before activating required-check changes.
- Keep the compatibility `test` check until the new `OpenPinch PR Gate` has
  been observed and separately authorized branch-protection migration verified.
- Verify trusted publishers and the `pypi` environment permit the new main
  dispatch context before any real publication.
- Metadata-only PR edits intentionally run no expensive tests and supply no
  new successful proof; reopen a ready PR or push a source change if a new
  gate is needed. This conservative behavior is documented.
- Expired/unavailable immutable artifacts stop recovery. Draft assets alone
  are not accepted as provenance; no silent rebuild fallback exists.
- Upload action output is bare hex; the shared output normalizes to the API's
  `sha256:` form. Pinned download-action source and official upload output
  documentation were checked during the audit.
