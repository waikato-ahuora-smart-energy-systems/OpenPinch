# Deployment and activation sequence

## Validation path

An event planner resolves the immutable candidate and required profile.
Shared validation either proves compatible reused lanes or executes them.
A top-level gate aggregates complete evidence. Develop and main pushes use
the same lane owners but never invoke publishing. Main PRs include solver
validation; a main push cannot inherit weaker develop-only evidence.

## Explicit release path

Version-preparation PR, then ordinary reviewed merge, then explicit release
dispatch. The dispatch resolves a main source commit and validates full-profile
proof. It produces or adopts a uniquely identified verified release bundle,
stages tag/draft assets, verifies TestPyPI, verifies PyPI, and only then
finalizes GitHub release publication. Each transition has manifest-bound
preconditions. Recovery reuses the original bundle and observes existing
state rather than rebuilding or blindly repeating writes.

## Safe rollout checklist

1. Implement and locally validate helpers, event/permission contracts, reusable
   lanes, explicit dispatch, and documentation. No remote settings changes.
2. Review the complete PR diff and run hosted checks on the actual candidate.
   Preserve `test` as a complete-gate compatibility check throughout migration.
3. Verify that merging the overhaul disables main-push publication; do not
   merge an intermediate state that retains an unintended auto-release path.
4. Observe the exact `OpenPinch PR Gate` check on a real PR. Test failure/skip
   propagation in local contracts and inspect hosted job dependencies/results.
5. Obtain separate authority to update required checks, preserve existing
   reviews and strictness, apply the change, and read the resulting rules.
   Do not remove the old requirement before the replacement is verified.
6. Inspect trusted-publisher records and the `pypi` environment's allowed refs
   and reviewer rules. Request any necessary configuration adjustment
   explicitly. Do not weaken approval requirements to make a dispatch run.
7. Confirm dispatch input validation and candidate/artifact checks without
   uploading packages. Record publishing as unverified until a separately
   authorized real release exercises its external boundaries.
8. Remove obsolete compatibility code only in a subsequent reviewed change.

## Failure and rollback boundaries

Before activation, revert repository changes through normal review if needed.
After required-check migration, any rollback must preserve an available
complete gate; do not leave an impossible required check or downgrade to an
ordinary-test-only gate. Any settings change needs specific authorization.

After an upload, there is no destructive rollback. Preserve original artifacts,
record which index is public, and use explicit resume. The existing staged
0.6.10 release requires its own verified legacy recovery procedure and user
authorization; this deployment does not automatically resume it.

## Verification layers

- Local: helper examples/properties, event matrix, permissions, workflow lint,
  selection/coverage preservation, artifact identity and recovery simulations.
- Hosted: real job names/dependencies, OS/solver lanes, report retention and
  immutable build evidence. No local test claim substitutes for this layer.
- Configuration: required checks, environment restrictions, publisher records.
- Publication: separately authorized release and external hash verification.

Completion reporting must identify which layers have evidence and which remain
pending. No remote rollout or publishing has been performed in this design.
