# HPR derived-case API implementation

The public sequence now selects a solved scalar HPR target, allocates its residual
with direct/all integration, returns an already solved optimized utility case,
and copies the utility definitions into a fresh receiver study when requested.
Original HPR numerical fixes remain intact.

## Unit 1: Selection and contracts

Application targeting provenance owns scalar zone/period inference. The HPR
resolver reuses retained target integrity and ownership checks in hpr_selection.
New consumers accept HPR targets only; current source fingerprints, local owners,
and scalar periods are required. The result owner remains the derived case.
Unsupported bases, conflicting scope/periods and thermal overrides fail before
source publication. Existing consumer compatibility is preserved.

## Unit 2: Case derivation

PinchProblem.residual_utility reuses the existing frozen numerical handoff.
PinchProblem.with_utilities_from validates a canonical replacement, aligns period
arrays by ID and binds implicit quantities when source and receiver unit defaults
differ. Segments, profiles, capacity metadata and active flags survive. Runtime
components use a connected graph clone with independent ownership and preserved
memberships. All analysis caches are cleared. The unreleased target-level
residual factory and catalog entry were removed; no alias is retained.

## Unit 3: Allocation shortcuts

Direct and all integration resolve a base before entering any source transaction.
Residual all integration delegates to its existing supported direct allocator and
returns the normal TargetOutput. This deliberately reuses the residual guard
rather than broadening transactional access to analyses requiring process streams.
Ordinary all integration retains its true subzone default; residual/scalar HPR
calls infer false and reject explicit expansion.

## Unit 4: Solved placement

The existing optimizer and frozen candidate allocator are unchanged. A common
finalizer runs deterministic direct or aggregate allocation for exactly the
requested periods and zone, compares each winning level duty at relative 1e-5
and absolute 1e-4 tolerances, records period outputs, then attaches placement
evidence. Failed finalization never commits into the source case. No second
optimization or HPR solve occurs. Explicit re-targeting remains supported.

## Unit 5: Forwarding compatibility

Exergy, energy transfer and all cogeneration variants share scalar selection
validation. Their established aggregate/subzone behavior is preserved; a first
regression caught and corrected an overbroad scope guard. Cross-case and
all-period scalar base broadcasting is rejected before execution. No-base batch
workflows retain their existing publication behavior.

## Unit 6: Documentation

Notebooks 08 and 19, their generator, HPR and placement guides, problem/workspace
API pages and inventories describe return types, ownership, period rules and
input-only JSON. Notebook 19 retains the user's 100/100/20/10 search controls.
Registration through workspace.add still copies inputs; the notebook keeps solved
placement objects for immediate observation. All 19 sources match the generator
without rewriting unrelated saved outputs. Public inventory: 201 operations.

## Validation and limitations

See ../../build-and-test/hpr-derived-case-api/build-and-test-summary.md for final
integrated outcomes. Regressions cover manual/shortcut equivalence, real placement,
component membership, capacity and segment retention, unit defaults, period
reordering, finalization failure, scope guards and observation without solves.
Hypothesis generates period permutations, units, active/capacity combinations and
operation sequences including empty sequences; the reference canonical input must
remain unchanged after every operation. Seed 20260715 is retained.

Utility transfer requires matching period sets. A scalar HPR handle cannot be
broadcast into multiple studies. Copying utilities does not install HPR equipment
or make donor residual results valid on the receiver. Ordinary JSON reloads start
unsolved. Separate area/cost, HPR-background and aggregate-basis extensions remain
outside this delivery. No commits, pushes or deployments were performed.

## Follow-up change audit

Three audit passes tightened target metadata integrity, frozen residual input
validation, zone/weight and fluid/pressure/enthalpy preservation, missing-period
capacity handling and singleton period vectors during unit conversion. Notebook
19 guidance and its generator were corrected. See change-audit.md for the
reproduced defects, 17 regression cases and final verification evidence.
