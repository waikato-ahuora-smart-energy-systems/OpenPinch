# Delivery logic and state transitions

## Validation and release event model

| Intent | Evaluation | Side effects |
|---|---|---|
| Develop change | Shared integration profile for immutable event SHA | Reports and CI artifacts only. |
| Ready PR to develop | Evaluate merge candidate using integration profile | Reports only; never version-write or publish. |
| Ready PR to main | Evaluate merge candidate using full promotion profile, including solver validation | Reports only; aggregate gate required. |
| Ordinary main merge | Validate resulting immutable commit | Reports/artifacts only; unchanged version is allowed. |
| Version preparation | Deliberately choose version and commit package/lock changes through a normal reviewed PR | Version changes precede validation; no publication. |
| Explicit release | Select a validated main commit and its committed version | Build/verify immutable bundle, then authorized publishing stages. |
| Explicit recovery | Load original release identity and bundle | Revalidate checkpoints; perform only missing authorized stages. |

The integration profile preserves current ordinary, docs, TESPy, performance,
optional-install, build, and cross-platform artifact lanes. The full promotion
profile adds the existing solver lane. A release cannot rely on integration
evidence alone. Exact trigger names and permissions are infrastructure design.

## Candidate validation

1. Resolve immutable candidate and required lane profile.
2. Check reusable evidence only for eligible trusted candidates. Require exact
   tree, compatible policy/toolchain/lock context, complete lane coverage, and
   the latest applicable successful attempt. A PR's merge tree must match the
   reused tree; matching source SHA alone is insufficient.
3. If evidence is absent, uncertain, incomplete, or inapplicable, execute the
   corresponding validation. A failed proof lookup is not successful proof.
4. Aggregate actual execution and accepted reuse per lane. Fail on any missing
   required proof. Emit candidate identity, evidence references, and reasons.
5. Treat a changed candidate/base as new validation, never carry the old gate
   across a different merge result.

## Release path

1. Requested: validate explicit request, main membership, committed version,
   and release policy. Reject an inconsistent existing version/tag identity.
2. Validated: require complete full-profile evidence for the source. Run missing
   validation rather than substituting a weaker profile.
3. Bundled: build once, record manifest, verify metadata and installed-artifact
   smoke tests. Retain the bundle for subsequent attempts.
4. Staged: prepare immutable tag/draft assets only after bundle verification.
   Existing exact state is acceptable; conflicting state is a terminal error.
5. Test index verified: inspect existing files, upload only as needed, then
   require exact complete visibility within the bounded deadline.
6. Production index verified: publish the same bundle with the same preflight
   and postflight rules, subject to the production publishing authorization.
7. Finalized: finalize the matching GitHub release only after both destinations
   verify. Recheck existing finalized state rather than republishing it.

This is a recoverable sequence, not an atomic cross-service transaction.
If production upload succeeds and GitHub finalization fails, report partial
completion explicitly and resume finalization using the original identity.
Never attempt to undo publication by deleting packages or moving tags.

## Recovery

Resolve the original source, run/attempt, manifest and artifact identity.
Verify retained bytes and external state. Resume the first unmet prerequisite;
matching completed checkpoints are no-ops. Conflicting remote bytes or source
identity stop recovery. If original bytes cannot be recovered from a verified
retained bundle or exact draft assets, stop with an actionable error; do not
rebuild under the same release identity. Supporting legacy 0.6.10 recovery is
a documented, separately authorized procedure, not automatic migration.

## Testable Properties

- Model-based state sequences never finalize before both index verifications.
- Repeating a completed transition with identical external state is idempotent.
- A conflicting artifact cannot reach a publish/finalize transition.
- Interrupted runs resume without changing the manifest or repeating an
  already satisfied upload need.
- Compare gate evaluation to a small reference predicate over required cells.
  Generate valid and adversarial evidence, including duplicate cells.

Example regressions complement generated tests: delayed visibility beyond
50 seconds; partial upload; failed finalization after production upload;
expired artifact; newer failed attempt; skipped performance lane; differing
merge tree; PR description edit; normal main merge with unchanged version.
