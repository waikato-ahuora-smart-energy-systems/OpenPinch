# Unit 2 Business Logic Model

## Scope

Unit 2 owns CoolProp capability preflight and reliable optimized search for
cascade, parallel, VC+MVR, and multiperiod HPR services. It consumes Unit 1's
budget, diagnostic, error, evaluation-mode, and detached-output contracts.

## Algorithms

### Prepared capability preflight

1. Reject invalid fluid identities and unsupported backends before even optional
   Carnot initialization; normalize topology fluids to actual stage counts.
2. Parse the deterministic warm-start state into evaporating/suction and
   condensing/discharge temperatures.
3. Resolve every VC refrigerant at its dew and bubble states.
4. Resolve every MVR fluid at its suction and discharge saturation states.
5. Identity/backend failure raises `HPRTargetingError` with bounded context and
   zero optimizer calls. State-probe failure at variable temperatures is recorded
   as an unsupported seed, not a rejection of the complete search space. The
   audit reproduced a valid alternative state for an unsupported R134a seed.

### Bounded warm-start-first search

1. Resolve one strict-positive `HPRSearchBudget`.
2. Normalize warm starts, then evaluate them in search mode before backend work.
3. Cache each exact finite coordinate tuple and retain every viable warm start.
4. Enforce the evaluation cap at the objective boundary, including warm starts,
   every restart and serial HPR-owned polishing. Pass iteration/evaluation
   controls to the generic optimizer as well; repeated points reuse the cache.
   Optional Carnot screening is a separate bounded search; final artifact
   reevaluations are outside the search cap and included in failure counts.
5. Merge backend, warm-start and the best sixteen visited viable candidates by
   exact coordinate, retaining the lower objective and deterministic ordering.
6. Final-evaluate ranked points. Return the first viable detached result.
7. If none succeeds, raise a bounded `HPRTargetingError`; unexpected raised
   exceptions are fatal and remain chained.

### Multiperiod behavior

All period cases receive the same resolved budget. Capability preflight covers
each period before the shared optimiser. Search cache identity remains the
shared design vector, and one final accepted vector is evaluated across all
periods before recursive detachment.

## Property Obligations

| ID | Category | Property |
|---|---|---|
| U2-P1 | Invariant | Every public budget is an exact positive integer pair |
| U2-P2 | Precedence | Named budget values override runtime values, which override defaults |
| U2-P3 | Easy verification | A preflight failure causes zero optimiser calls |
| U2-P4 | Invariant | Each configured VC and MVR stage appears exactly once in prepared capability facts |
| U2-P5 | Idempotence | Repeated exact search coordinates invoke the objective once |
| U2-P6 | Invariant | Warm starts are evaluated before the backend |
| U2-P7 | Metamorphic | Candidate ordering does not change the best unique finite point |
| U2-P8 | Model property | Candidate-local failures permit a later viable point |
| U2-P9 | Model property | Raised fatal defects abort immediately |
| U2-P10 | Invariant | Diagnostic counts cover all observed failures while representatives never exceed 16 |
| U2-P11 | Metamorphic | Backend budget exhaustion cannot discard a retained viable warm start |
| U2-P12 | Invariant | Single- and multiperiod paths carry identical budget values |

## Exclusions

No direct process-MVR stage changes, new fluid allowlist, wall-clock API,
thermodynamic-equation change, or generic-optimiser dependency on HPR is added.
