# Domain entities

## ReviewedCandidate

Repository identity; original PR number; fixed head branch develop and base
branch main; head SHA; base SHA; tested merge-tree SHA; aggregate review decision;
workflow identity and policy; required lane evidence. All identities come from
fresh repository APIs or locally verified Git objects, never PR text alone.
Use full hexadecimal Git object IDs and positive integer run/PR identifiers.

## PreparationRecord

Versioned strict JSON, proposed path `.github/release-preparation.json`:
schema, repository, original PR, reviewed head/base/tree, baseline version,
target version, allocation kind (patch or explicit), preparation identity,
review evidence references. Reject unknown keys, invalid types and oversized
records; maximum 64 KiB. No shell commands, credentials or user-supplied paths.

Preparation identity is SHA-256 of canonical schema/repository/PR/reviewed
head/base/baseline/target data. The record does not contain its own commit SHA;
that would be self-referential. Resolve the generated commit and bump PR through
Git/API proof, checking parent and exact allowed transformation.

One current record is stored in source. Previous records remain in Git history.
A record is a claim, not authority: recompute identity and verify external proof.

## LaneEvidence

Repository, workflow path, event, run ID, attempt, job ID/name, candidate head,
tested tree, policy, toolchain/matrix identity, conclusion and any parent evidence
reference. One successful concrete job per required matrix entry. Bounded reuse
graph (maximum 8 links, 100 nodes); cycles or missing leaves invalidate reuse.
The latest applicable run/attempt must not be hidden by older successful proof.

## ValidationDecision and ReleaseBundle

Per lane: execute or reuse with verified evidence and reason. A skipped Actions
job alone never fulfills a requirement. Summary lists fresh and reused lanes.

New release manifests bind preparation identity and the canonical evidence-plan
digest to the existing exact source, tree, build run/attempt, wheel/sdist hashes
and artifact identity. Evidence is embedded in the manifest, keeping the existing
three-file release bundle shape. Main gate and publisher independently validate
it. Policy/schema versions distinguish legacy full-validation proof from new
mixed fresh/reused proof; never reinterpret a legacy skipped job as evidence.

## External observations and state

Remote branch heads, PR status/reviews, tag/release identities, package-index
state and workflow jobs are observations with bounded I/O. Unknown is distinct
from absent. Stable preparation identity plus branch/PR history is the durable
retry record; workflow outputs are conveniences, not the source of truth.
