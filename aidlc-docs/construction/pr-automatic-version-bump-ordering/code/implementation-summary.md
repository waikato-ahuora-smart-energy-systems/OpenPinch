# Pull-Request Automatic Version-Bump Ordering Implementation

## Modified Delivery Behavior

- A same-repository, non-draft pull request targeting `main` receives one
  automatic semantic-version bump when its candidate version equals the base.
- Label selection preserves `major`, `minor`, then `patch` precedence; a title
  marker is the fallback and patch is the default.
- The pinned bump tool updates all three canonical version records and creates
  no tag. The job validates the lock entry and forward version before pushing.
- An already-forward version produces no commit. A behind version fails closed.
- Fork pull requests skip the write job and retain read-only validation.
- The release-version job depends on bump evaluation and checks out the latest
  named PR head rather than the workflow event's original SHA.
- Any later synchronize event or manual rerun sees the forward version and
  remains mutation-free; correctness does not depend on a recursive bot-push
  workflow event.
- The aggregate gate distinguishes same-repository main PRs, fork main PRs, and
  develop PRs when enforcing bump and release job outcomes.

## Tests and Documentation

- Replaced the former no-mutation assertion with static contracts for scoped
  permissions, deterministic bump policy, no-tag behavior, updated-head
  validation, dependency ordering, fork behavior, and aggregate-gate wiring.
- Extended release documentation consistency tests.
- Updated README and the Read the Docs developer release procedure.

## Property-Based Testing

PBT-01 was assessed for convergence, but fixed workflow event scenarios provide
a clearer executable oracle. PBT-02 through PBT-10 are not applicable because
the correction introduces no domain data generator, numerical algorithm,
serialization contract, or persistent application state.
