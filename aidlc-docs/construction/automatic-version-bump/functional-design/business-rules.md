# Business rules and testable properties

## Version allocation

Parse strict non-negative canonical major.minor.patch tuples. All three metadata
locations must agree. Baseline is the version at the recorded main base commit,
not the GitHub Latest pointer. Existing tags/index observations are collision
checks, not proof that source was validated. Legacy 0.6.10 is occupied but does
not prevent preparing 0.6.11; it is not repaired by this workflow.

If reviewed develop equals baseline, propose baseline.patch + 1. If reviewed
develop explicitly increases major/minor, preserve that reviewed version.
An existing verified preparation retains its target. Other unexplained version
changes, downgrades, conflicting tags/index entries or inconsistent metadata
block for diagnosis; never skip over an interrupted release by allocating again.
Check target availability on both indexes and GitHub before preparation and
again at publication. API errors never mean absence.

Explicit major/minor candidates still need a preparation record and normal
merge protections, but receive no additional version increment. Their generated
preparation PR may contain only the record.

## Exact allowed transformation

Permit only project.version in pyproject.toml, tool.bumpversion.current_version
in .bumpversion.toml, and the unique local openpinch package version in uv.lock,
plus the strictly validated preparation record. Compare parsed values and exact
remaining bytes/tree paths; dependency edits or broad TOML reformatting do not
qualify as version-only equivalence. Reject duplicate keys, ambiguous package
entries, symlinks/mode changes or extra files. Version edit helpers must not
execute bumpversion hooks, build backends or candidate scripts in the write job.

## Lane reuse policy

Candidate reusable lanes: test, hpr-tespy-tests, performance-tests, solver-tests
and workflow-lint. Each needs concrete successful compatible review evidence,
unchanged tested source/configuration/toolchain apart from the proven version
transformation, and no version-dependent behavior in its selected tests.
Code Generation must audit selected tests for version dependencies; any affected
checks are run fresh, never silently omitted to make a lane reusable.

Always fresh on the bumped main commit: policy/proof verification, docs (version
rendering), optional-install-smoke surfaces, artifact-build, artifact-install-smoke
on all existing operating systems, and artifact-install-tespy-smoke. No matrix
entry or existing quality threshold is removed. Artifact packaging regressions
that overlap a reusable test lane must still run fresh as package validation.

Use the same proof machinery on the bump PR, updated original PR and develop
push to avoid shifting duplicate full suites earlier in the flow. Direct develop
validation without verified preparation evidence retains normal behavior.
Fresh builds must run even when upstream heavy test jobs are legitimately
skipped; model Actions dependency conditions explicitly in workflow tests.

Proof must establish the tested PR merge tree, not just the PR head SHA. Main's
actual tree must equal that tested tree after the exact allowed transformation.
An unrelated main change, dependency change, policy change or unavailable source
tree invalidates reuse. Missing evidence reruns the relevant lanes (full suite
when impact cannot be proven). Never hide a later failing attempt behind a
historical green result. Review proof reuse does not substitute human approval.

New policy is delivery-v2. Legacy delivery-v1 recovery retains its existing
all-jobs-success verifier. The v2 manifest binds fresh and reused evidence, and
both main gate and publisher recompute it; arbitrary Boolean reuse flags cannot
authorize publishing. Evidence deletion/expiry before publishing blocks if it
cannot be verified; a new full validation can create replacement proof before
first publication. After publication starts, use original-bundle recovery only.

## Testable Properties (PBT-01)

- Version parser/record codec: round-trip canonical structured inputs.
- Allocation: invariant target exceeds baseline; explicit major/minor preserved;
  independent tuple-based oracle returns allocate, reuse or block.
- Preparation: idempotence across duplicate notifications and partial writes;
  at most one branch/PR identity per request. Conflicts never overwrite data.
- Diff verifier: invariant any mutation outside exact allowed fields rejects
  reuse. Generate structured TOML/version edits and adversarial path/mode edits.
- Evidence verifier: invariant no missing/failed/duplicate job or altered
  repository/tree/policy/matrix authorizes skipping. Independent set-coverage
  oracle compares required lanes with successful concrete leaves.
- State coordinator: stateful model with review, test completion, base/source
  change, branch creation, PR creation/merge/closure, publication and retry.
  Assert after each step: no publish without proof, no duplicate bump, no user
  branch overwrite, and no mutation in blocked states. Include empty sequences.
- Event ordering: review then tests or tests then review reaches equivalent
  readiness when observations are otherwise identical (commutativity).
- Release identity: invariant emitted bundle source/version equals validated
  bumped source; repeats preserve original artifact identity.

Use bounded domain-specific Hypothesis strategies for versions, records, Git
graphs, matrix jobs and event sequences. Keep shrinking enabled and fixed CI
seed 20260715. Pure reference models must not call production decision helpers.
External I/O is faked; real parsers, decisions and state adapters execute.

Concrete regressions complement properties: 0.6.10 to 0.6.11; explicit minor;
tests-before-review; review-before-tests; reapproval after bump; stale base;
hidden dependency edit in uv.lock; changed build settings in pyproject.toml;
cancelled latest run; missing solver proof; skipped required job; branch-created
PR-create interruption; closed bump PR; concurrent source advance; two review
PRs; repeated bump merge; main build after skipped heavy jobs; partial publication
resume; unavailable original evidence. No live publication in tests.

## Extension compliance

PBT-01 compliant: component properties/categories above. PBT-02 through PBT-10
implementation verification N/A at Functional Design; round trips, invariants,
idempotence, oracle, stateful sequences, structured generators, reproducibility,
existing Hypothesis and explicit regressions are specified for Code Generation.
Security and Resiliency extensions disabled; explicit delivery safeguards remain
mandatory. Frontend properties N/A: no user-interface component.
