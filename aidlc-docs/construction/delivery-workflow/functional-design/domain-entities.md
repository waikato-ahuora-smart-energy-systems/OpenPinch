# Delivery entities

These are logical contracts for delivery helpers and workflow inputs, not new
OpenPinch application models or a new persistent database.

| Entity | Required identity and data | Invariant |
|---|---|---|
| Candidate | Repository, event, source SHA, evaluated SHA/tree, base SHA where relevant, policy revision | Evidence always identifies the evaluated tree; a branch name alone is not identity. |
| ValidationProfile | Named lanes, required expanded matrix cells, permitted exclusions, toolchain/lock context | Every required cell has an explicit owner; exclusions are policy decisions, not inferred from skipped jobs. |
| ValidationEvidence | Candidate, profile/policy identity, run ID and attempt, per-cell conclusion, provenance | Evidence must be complete, current, trusted, and applicable to the candidate. |
| GateDecision | Pass/fail, evaluated candidate, executed or reused proof per lane, diagnostic reasons | Pass requires proof for every mandatory lane. |
| ReleaseRequest | Repository, immutable main commit, intended version, deliberate initiating event | Ordinary pushes and PR metadata changes cannot create this entity. |
| ArtifactManifest | Source commit/tree, version, build run/attempt, artifact ID/digest, exact filenames and SHA-256 hashes | All promotions and recoveries reference one immutable bundle. |
| IndexObservation | Destination, version, observation time, filenames/hashes or typed error | Absent/partial is distinguishable from conflict and transport failure. |
| ReleaseProgress | Manifest identity, verified destination checkpoints, tag/release identity | Checkpoints are revalidated against external facts; prior green labels alone are insufficient. |

The expected distribution set remains exactly one universal wheel and one
source archive for the selected version. Manifest/checksum sidecars are
evidence, not extra package-index distributions. A resume request references
the original manifest; it does not implicitly create a new build.

## Testable Properties

- Candidate/proof matching is an invariant under reordering of job records.
- Removing a mandatory successful cell cannot improve a gate decision.
- Changing candidate identity, policy, provenance, or a required conclusion
  invalidates proof unless a separately defined exact-tree reuse rule applies.
- Manifest serialization has a structural round-trip property if a new
  serializer/parser pair is introduced; duplicate identities must be rejected.
- No application-domain properties are introduced; numerical test contracts
  remain unchanged.
