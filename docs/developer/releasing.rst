Automatic Releases and Recovery
===============================

Ordinary merges and release preparation
---------------------------------------------

Develop and main pushes, and ready pull requests, run shared validation.
Normal review and tests run first on the existing ``develop`` to ``main`` PR.
Once the current review is approved and its complete validation lanes pass,
**Prepare Release** opens a version-only PR into ``develop``. The default is
one patch increment; a reviewed explicit major/minor increase is preserved.
The coordinator updates ``pyproject.toml``, ``.bumpversion.toml`` and ``uv.lock``
and records the reviewed source/evidence identity. It never approves or merges.

Merge the bump PR into develop, then merge the original updated PR into main.
Renew review approval if branch protections require it. The complete main-PR
gate deliberately waits for version preparation; this does not prevent the
coordinator from recognizing successful review tests. PR validation remains
read-only; source writes occur only in the trusted main-context coordinator.

Main reuses independently verified review evidence for unaffected expensive
lanes and general tests. Packaging/version-sensitive tests, documentation,
dependency-surface checks, builds and distribution-install matrices run fresh.
Coverage remains enforced by the original full review run. Missing, expired,
failed or incompatible evidence causes normal validation instead of unchecked
skips. Workflow/policy changes cannot reuse evidence across the policy change.
The publisher independently rechecks the evidence and the exact new artifacts.

Activation and blocked preparation
----------------------------------------

Install the coordinator and validation changes together through normal review.
Only a base predating the coordinator qualifies for the one-time bootstrap
exception; there is no label or input that bypasses preparation afterward.
No release is automatically repaired during bootstrap (including legacy 0.6.10).
Repository policy must permit Actions to create PRs. GitHub may require a
maintainer to approve bot-created PR workflow runs. Current aggregate review
state must be ``APPROVED``; configure required reviews through separately
authorized repository settings if it is unavailable. No personal token, GitHub
App credential or protection bypass is installed by this implementation.

The trusted coordinator reconciles after review notifications and PR validation,
regardless of which finishes first. Manual **Prepare Release** reconciliation
performs the same checks. Duplicate events reuse the existing branch/PR; an
interrupted PR-create step can resume from its existing branch. A closed or
conflicting preparation requires maintainer intervention. New changes before
the bump merges invalidate that bump PR: close the stale PR, then reconcile
the newly reviewed candidate. Never force-push over conflicting branch content.
If changes arrive after the bump merges, retain its unpublished version and
rerun normal validation when exact review equivalence no longer holds.

Completed title/body-only edits are excluded from review-evidence selection
only when the explicit metadata marker and planning job succeeded in the current
run attempt, with no executed validation lanes. Newer failed, cancelled,
incomplete or unclassified validation runs still block older evidence. Historical
metadata runs without the marker require a fresh candidate validation. This
does not make metadata-only events satisfy the merge gate.

Preparation currently requires concrete full review-lane evidence, so the
older develop-to-main aggregate-only optimization is disabled for this path.
After preparation, verified reuse applies to updated PR, develop and main runs.

Automatic new release
---------------------

After ``CI Main`` completes successfully, ``ci-publish.yml`` (Release and
Publish) starts as a separate workflow. It verifies the source repository,
main push, workflow path, commit, full validation evidence and original build
artifact. It promotes those exact tested bytes without rebuilding or running
validation again. The listener's checkout supplies trusted automation code;
the original source run, not the listener's default-branch SHA, identifies the
release candidate.

A failed or cancelled main run cannot publish. Existing completed versions
are verified against their original manifest, retained build artifact, tag,
GitHub assets and both package indexes before a successful no-op is reported.
A later same-version commit is not a new release and its newly built files
are not substituted for the original release. Drafts, partial publication,
prereleases, conflicts or missing/expired proof block automatic publication
with recovery guidance. A tag alone does not establish completion.

Because this completion proof requires the original CI artifact, automatic
no-op verification also stops after that artifact expires. Prepare a reviewed
new version or investigate the original evidence; do not bypass the proof.
Legacy 0.6.10 artifacts predate this contract and require the separate recovery
procedure below. Restoring automation does not repair that release implicitly.

Manual new release
------------------

After publisher/environment compatibility has been verified, a maintainer
may also run ``ci-publish.yml`` (Release and Publish) on ``main`` with:

* ``mode``: ``new``
* ``version``: the exact committed ``X.Y.Z`` version
* all source artifact inputs empty

The dispatch evaluates its immutable main commit, runs the full validation
profile, and builds one bundle. It records a release manifest and immutable
build artifact ID, digest, run ID, and attempt. The same verified distributions
are installed in the artifact smoke jobs and promoted without rebuilding.

It creates or verifies an annotated tag and draft GitHub release, uploads to
TestPyPI, verifies the exact filenames and SHA-256 hashes, and then reaches the
protected ``pypi`` environment for production publication. After PyPI succeeds
and both destinations verify, the workflow publishes the GitHub release.
No public release is declared complete merely because an upload returned 200.
Existing prereleases (draft or public) are rejected rather than silently
accepted as stable releases. Recovery leaves Latest selection to GitHub;
it never explicitly promotes an older resumed release to Latest.

Verification uses a five-minute default visibility budget per destination.
Absence, matching partial visibility, and transient transport errors may retry;
wrong hashes, unexpected files, malformed responses, or access failures do not.
The logical deadline bounds retries; a separate job timeout covers blocked
underlying I/O. Progress is on stderr and machine-readable status on stdout.

Interrupted releases
--------------------

Use **Re-run failed jobs** when the original build and request remain valid.
Cross-job downloads use immutable artifact IDs, not the retry's new attempt
number. Do not rerun all jobs after staging a version: a new request cannot
rebuild an existing tag. To resume in a separate run, choose ``mode=resume``
and provide the original ``version``, ``source_run_id``,
``source_artifact_id``, and ``source_artifact_digest`` from the bundle summary.
Use the summary's digest including its ``sha256:`` prefix. Validation retries
use the latest job evidence from the original source run while retaining the
original build attempt and bytes; a newer failed lane blocks recovery.

Resume revalidates the original full-profile source evidence, artifact
provenance, manifest, and exact bytes. A complete index is a no-op; partial
uploads can finish; conflicting state stops. A public package followed by a
failed GitHub finalization remains partially completed, not rolled back.
Never delete packages, move tags, or overwrite assets to resolve a conflict.

CI requests 30-day retention for reports and recovery bundles, subject to
repository limits. Draft assets retain matching bytes, but are not alone proof
of an expired artifact's trusted provenance. This implementation stops if the
original immutable artifact/evidence cannot be verified. Preserve the source
run and bundle before expiry; do not rebuild under the same version as fallback.

Automatic and manual publishing are serialized across the repository in the
same non-cancelling workflow lock. Active publication is not
cancelled by a newer request. Pending requests are not a guaranteed FIFO queue;
resubmit a cancelled pending request explicitly if still required.

Legacy 0.6.10 recovery
---------------------------------------------

The failed legacy run is ``36132464168`` at source ``2e743566``. Its bundles
predate the new manifest contract and cannot be fed into the new resume mode.
Recovery requires separate maintainer authorization and verification of the
original artifact ID/digest, source run, draft assets, exact index hashes, and
production publisher/environment configuration. Read its existing job status
before deciding whether re-running only its failed verification is appropriate.
The legacy run may then continue into production publication; do not trigger
that continuation as a diagnostic action. If original evidence is unavailable,
stop and prepare a new version through the explicit process instead.

Required-check and publisher activation
---------------------------------------------

Deploy and observe ``OpenPinch PR Gate`` on a real PR before changing branch
protection. The compatibility check ``test`` depends on that complete gate.
With separate authorization, require the observed new check while preserving
strict up-to-date checks and review protections; read back the actual rules.
Remove the old requirement/alias only after the replacement is active.

The publishing filename remains ``ci-publish.yml`` and the production
environment remains ``pypi``. Inspect both package-index trusted publishers
and environment deployment-ref restrictions before enabling publication from
main rather than the legacy tag context. Preserving names does not prove
configuration compatibility. Do not disable reviewers or broaden credentials
to bypass an incompatible setting.

Validation evidence
-------------------

The shared validator emits JUnit reports, test durations, and candidate/policy
summaries. Test and coverage failures remain failures; retries do not conceal
numerical regressions. The local suite, hosted OS/solver matrix, actual required
checks, and a real authorized publication are distinct verification layers.
