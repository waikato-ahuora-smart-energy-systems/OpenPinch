Explicit Releases and Recovery
==============================

Ordinary merges and release preparation
---------------------------------------------

Develop and main pushes, and ready pull requests, run shared validation.
Neither a normal main merge nor a PR label/title changes the package version
or publishes distributions. Prepare a version increase in a normal branch,
update ``pyproject.toml``, ``.bumpversion.toml``, and ``uv.lock`` together, and
review the diff before committing. Existing bump-my-version configuration may
be used with ``--no-tag``; the release workflow owns annotated tags.

Run ``uv lock`` and ``uv run --no-sync python scripts/check_lockfile_version.py``
after preparing the version. Merge only after the complete PR gate and existing
review requirements pass. Source code changes are never made by PR validation.

New release
-----------

After publisher/environment compatibility has been verified, a maintainer
deliberately runs ``ci-publish.yml`` (Explicit Release) on ``main`` with:

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

Publishing is serialized across the repository. Active publication is not
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
