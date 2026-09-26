# Delivery infrastructure design

## Workflow ownership

| Owner | Trigger and role | Authority |
|---|---|---|
| `ci-validation.yml` (new) | Reusable `workflow_call`; explicit candidate/profile inputs; owns shared lanes and evidence output | Contents read; actions read only where evidence lookup needs it; no publishing secrets. |
| `ci-develop.yml` | Develop pushes; thin integration-profile caller | Read-only validation. |
| `ci-pull-request.yml` | PRs targeting develop/main; thin candidate planner, validation caller, and stable top-level gate | Read-only validation; remove version-writing job. |
| `ci-main.yml` (new) | Main pushes; full-profile validation without version bump or publication | Read-only validation and build/report artifacts. |
| `ci-publish.yml` | Explicit dispatch only; new release or resume using immutable source/artifact identity | Per-job scoped permissions as below. Preserve publishing filename. |

Version preparation is a documented local/script operation on a normal PR
branch, not an automatic branch-writing PR job. It selects and commits the
version and lockfile before validation. Publication is a separate explicit
dispatch for a validated main commit. No new PAT or bot credential is needed.

Use local reusable workflow references from the caller revision, not a moving
remote branch reference. GitHub supports same-revision local workflow calls;
called workflows cannot elevate the caller's permissions. Keep privileged
publication outside the read-only reusable validator.
[GitHub reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)

## PR event and gate behavior

Handle opened, synchronize, reopened, ready-for-review, converted-to-draft,
and edited events. A lightweight planner distinguishes base-branch edits from
description/title-only edits. Base changes select and validate the new merge
candidate; description-only edits start no expensive lanes. Label changes no
longer trigger version changes or determine publication.

The top-level stable check is proposed as `OpenPinch PR Gate`. It must not
report success merely because lanes were deliberately not run. Description
edits can only reassert success with compatible existing candidate/profile
proof; otherwise report a non-successful actionable result without starting
the expensive profile. Use a separate concurrency group for metadata-only
events so they cannot cancel an active candidate-validation run.

During migration retain a top-level `test` compatibility check depending on
the complete new gate, not merely the ordinary test lane. After actual remote
rules require the observed new check, remove the compatibility alias in a
separate reviewed change. The observed check name, not a guessed nested
reusable-workflow name, is the migration target.

Draft PRs are not validated as merge-ready. Reopening/ready-for-review causes
fresh planning. Candidate changes invalidate evidence. No `pull_request_target`
execution of untrusted source and no merge-queue trigger are introduced.

## Jobs, permissions, and timeouts

| Job responsibility | Permissions above read-only contents | Bound |
|---|---|---|
| Evidence lookup | Actions read | Explicit API/subprocess budget; bounded pagination. |
| Ordinary validation | None | Preserve existing 45-minute outer limit initially. |
| Docs, optional install, TESPy | None | Preserve existing 20-minute limits initially. |
| Performance | None | Preserve bounded command timeout and a larger job bound. |
| Solver | None | Retain current 120-minute outer bound until hosted measurements justify reduction. |
| Artifact build/smoke | None | Preserve existing finite bounds; isolate OS matrix cells. |
| Tag/draft staging and finalization | Contents write | Five-minute job bound; no source-branch writes. |
| TestPyPI upload | ID-token write | Ten-minute job bound; download and verify artifacts, no arbitrary source build. |
| PyPI upload | ID-token write; existing `pypi` environment | Ten-minute job bound; same verified bytes. |
| Package-index pre/postflight | None | Fifteen-minute job bound, covering setup plus maximum 600-second logical polling budget. |

Keep Python/uv versions and pinned action references initially unchanged.
Retain the solver's Ubuntu 22.04 runner and existing cross-platform smoke
matrix. Runner-version migrations are not mixed with this delivery refactor.
Any future runner changes require independent evidence.

PR and branch CI cancel superseded candidate runs within distinct groups.
Explicit publishing and recovery share a repository release concurrency group
with cancellation disabled. Repository-wide serialization is intentionally
conservative and also avoids competing latest-release updates across versions.
Do not rely on FIFO or on every pending request surviving: GitHub concurrency
can replace pending work. A cancelled pending request has performed no release
mutation and must be explicitly resubmitted if still wanted.
[GitHub concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)

## Storage and reports

Retain JUnit/duration reports and release bundles for a requested 30 days,
subject to repository limits. Artifact names include source run and attempt;
consumption uses immutable artifact identity plus manifest verification.
Only the exact wheel and sdist are passed to publishing, never report files.
Draft assets hold matching bundles/checksums and trusted manifest evidence.
Recovery verifies origin and hashes before trusting those assets.

Do not make privileged jobs restore caches produced by untrusted PR jobs.
Publishing jobs need no dependency cache or build environment. Reports upload
on failure when available; summaries distinguish setup failure from missing
test evidence. Secrets never enter uploaded reports.

## Publishing environment compatibility

Retain `ci-publish.yml` and the production `pypi` environment name to minimize
publisher migration. Existing YAML has no TestPyPI environment declaration;
do not add one without inspecting its configured publisher constraints.
Remove automatic self-dispatch and `actions: write` if the explicit workflow
can complete the full release chain in one run. Finalize GitHub only after
production verification succeeds.

Explicit release requests execute trusted workflow code from main, resolve
and pin the requested source commit, verify main ancestry and the committed
version, and reject branch/tag dispatch contexts outside the approved entry
policy. Resume likewise uses trusted current release tooling against the
original artifact identity, not arbitrary helper code from an old artifact.
Verify actual environment deployment-ref restrictions and trusted-publisher
configuration before activation: retaining a filename is not proof that
moving production execution from a tag to main is permitted.

## Compliance

No new cloud, network, database, or messaging infrastructure is required.
N01-N18 are mapped through job bounds, read-only validation, manifest-bound
publishing, serialized mutations, retained reports, and staged migration.
Property-based obligations remain in functional/NFR design; no implementation
test obligation is due in this infrastructure-only stage. Security and
Resiliency extensions remain disabled; explicit security requirements apply.
