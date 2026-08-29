# Unit 2 Business Logic Model

## Design Outcome

Unit 2 accepts a normalized Unit 1 request/model plus detached direct or Total
Site period snapshots and returns a complete `UtilityPlacementResult`. It owns
candidate replay, duty allocation, thermodynamic and monetary evaluation,
cogeneration adaptation, weighted aggregation, bounded optimization, feasible
alternative selection, and typed operational failures. It does not expose a
problem/workspace method, mutate a canonical `Zone`, or implement a new target,
turbine, or optimizer algorithm.

All nine Functional Design decisions select option A. The design therefore:

- extends `UtilityPlacementOptions` test-first to restore the approved FR-011
  optimizer method and override contract;
- reuses existing targeting on candidate-local data reconstructed from an
  immutable snapshot;
- aligns residual process and allocated utility profiles on their union
  temperature grid;
- always computes thermodynamic evidence and computes monetary evidence only
  for monetary requests;
- inherits non-placement turbine settings from explicit arguments and then the
  problem configuration;
- distinguishes candidate-correctable turbine failures from run-level adapter
  or configuration failures;
- separates feasible and infeasible backend values with a deterministic finite
  monotone scalar transform;
- memoizes exact normalized coordinate tuples within one run; and
- verifies tiny analytical cases against a deterministic structured grid.

## Contract Reconciliation Before Numerical Work

The approved Unit 1 design and FR-011 require an overridable method plus
validated backend options. The Unit 2 contract exposes:

- `method`, defaulting to `cmaes` and limited to the existing
  `OptimisationMethod` values;
- `run_count`, a positive integer;
- `cluster_tolerance`, finite and non-negative;
- `local_method`, a non-empty string accepted by the existing service; and
- sorted immutable `backend_options` containing JSON-safe values.

Existing fields map as follows: `iteration_limit` to `maxiter`,
`evaluation_limit` to `maxfun`, `candidate_limit` to `max_minima`, and `seed`
to `seed`. Method-specific names and value rules are validated by the existing
optimizer service before any candidate targeting. No package-root export is
added.

## End-to-End Service Flow

### 1. Prepare detached period snapshots

The Unit 3 caller will resolve the selected zone, scope, canonical period IDs,
weights, and explicit turbine overrides. Unit 2 receives an isolated zone copy
or equivalent detached source. For each period in caller order, the context
builder runs or reuses the selected direct or Total Site base target on that
copy, then extracts immutable data:

- shifted and real interval temperatures and enthalpy/load profiles;
- residual hot and cold utility requirements;
- process interval heat-capacity/duty evidence needed for entropy;
- approach-temperature and utility-feasibility limits;
- ambient absolute temperature;
- period identity and raw non-negative weight; and
- resolved target identity and units.

Mutable problem tables, streams, targets, and zones are discarded at the
boundary. Candidate replay reconstructs fresh local problem tables and utility
streams from the snapshots, so repeated evaluation cannot accumulate duty or
target state.

### 2. Build the Unit 1 placement model

The builder first produces the complete keyed `PlacementFeasibilityEnvelope`.
Unit 1 intersects its per-period coordinate bounds, applies caller narrowing
and ordering propagation, and returns the immutable `UtilityPlacementModel`.
An empty intersection fails before backend construction.

### 3. Construct the evaluation session

One service call creates a parent evaluation session containing the model,
context, resolved turbine settings, positive objective scale, and an initially
empty exact-coordinate memo. A multi-process backend may create isolated,
pickle-safe worker copies with independent memos. Sessions exist only for the
call and are not stored on the problem, workspace, or module.

Coordinates are normalized to finite Python floats with signed zero removed.
The exact normalized tuple is the memo key. Within one process/session, an
identical tuple returns the same immutable compact evaluation without targeting
again; a merely nearby tuple remains a distinct physical candidate. Duplicate
physical evaluation across isolated worker processes is permitted because
sharing a mutable memo would break backend isolation and pickling.

### 4. Verify and decode one candidate

The callback delegates vector verification and decoding to Unit 1. Bound,
temperature-direction, positive-Kelvin, ordering, separation, and fixed-span
failures become an infeasible evaluation with ordered diagnostics. They do not
throw through the black-box backend.

### 5. Replay the candidate in every period

For each selected period in canonical request order:

1. reconstruct fresh local shifted/real problem-table data;
2. construct ordered hot and cold utility streams from the decoded levels;
3. invoke the existing direct or Total Site utility-targeting adapter;
4. capture each level's assigned duty and the real-temperature process
   intervals that duty serves;
5. verify full hot and cold coverage; and
6. calculate period objective evidence.

The same decoded temperatures and template identities are used in every
period. A zero-weight period is still replayed and must be feasible. Evaluation
stops after the first blocking period failure, but the diagnostics retain that
period and all evidence already obtained.

### 6. Verify duty coverage

For side `s` and period `p`, let `R[s,p]` be the non-negative residual demand
from the base target and `A[s,p]` the sum of non-negative assigned duties. The
signed residual is `E[s,p] = A[s,p] - R[s,p]`. Coverage passes exactly when
`abs(E[s,p]) <= coverage_tolerance` for both sides.

No residual is silently clipped before this comparison. After a passing
comparison, values within tolerance may be normalized to positive zero in the
public result. A level may receive zero duty and remains present in declared
order.

### 7. Calculate thermodynamic evidence

The evaluator extracts the real-temperature process hot/cold composites,
inserts the active allocated candidate utility streams into their union
temperature grid, and uses the existing cascade to form candidate-local
balanced composite curves. The hot and cold balanced curves are then aligned
on their union heat-load grid.

For each matched sensible heat-load interval `j`:

`delta_S_j = CP_j * ln(T_out,j / T_in,j)`.

An isothermal interval uses the signed limit `Q_j / T_j`. Hot-composite and
cold-composite terms are added with stable finite summation. This ranks a
utility temperature closer to its matched process temperature ahead of a
farther one. Inactive utility coordinates do not alter the balanced curves.

The reported utility and process terms retain the public decomposition identity
and sum to total balanced-composite entropy generation. Values within numerical
noise may normalize to zero. Exergy destruction is ambient absolute
temperature times physical entropy generation.

Before thermodynamic ranking, positive duty assigned to generated fallback
`HU` or `CU` levels produces a deterministic infeasible penalty and diagnostic.

Thermodynamic evidence is computed for every feasible period in both objective
modes.

### 8. Calculate monetary and cogeneration evidence

Monetary mode computes thermal purchase cost as the canonical sum over all
levels:

`thermal_cost = sum((allocated_duty_kW / 1000) * price_per_MWh)`.

Only eligible hot levels with positive assigned duty are forwarded, in
descending temperature order, to a fresh `MultiStageSteamTurbine`. The adapter
uses `above_pinch` mode. Explicit placement arguments override the existing
problem power/turbine settings; omitted values inherit those settings.
Candidate temperatures and duties are the only optimized turbine inputs.

No eligible positive-duty level produces zero work without invoking the
turbine. Otherwise, cogenerated work comes from the detached solve result,
electricity credit is `(work_kW / 1000) * electricity_price_per_MWh`, and net
monetary objective is thermal purchase cost minus electricity credit.

A candidate temperature/duty combination that is physically incompatible with
otherwise valid turbine settings is ordinary candidate infeasibility. Missing,
invalid, unsupported, or internally inconsistent settings, adapter contract
violations, and unexpected turbine failures are typed run-level errors because
changing placement cannot reliably correct them.

### 9. Aggregate periods

For selected period cost `C[p]` and raw weight `w[p]`, the aggregate objective
is the canonical ordered finite sum `sum(w[p] * C[p])`. Weights are not
normalized. A zero-weight period contributes zero but remains mandatory for
feasibility and reporting.

Thermodynamic aggregate evidence is always retained. Monetary aggregate
evidence is retained only for monetary mode. Period results remain in selected
canonical order regardless of the mathematical commutativity of the sum.

### 10. Map feasibility to a backend scalar

The backend must receive one finite scalar even for expected infeasibility, and
no infeasible point may outrank a feasible point. A positive context-derived
scale `S` conditions the selected physical objective `C`. Feasible values use
the strictly increasing transform:

`F(C) = 0.5 + atan(C / S) / pi`, which lies strictly between zero and one.

For normalized total violation magnitude `V >= 0`, infeasible values use:

`P(V) = 1 + V / (1 + V)`, which lies from one through but below two.

Thus every finite feasible objective ranks ahead of every infeasible penalty,
including negative net monetary objectives, while ordering among feasible
objectives is unchanged. Diagnostics and physical objectives remain in the
memo; transformed values never enter public result contracts.

### 11. Run bounded optimization

The coordinator constructs the existing `OptimisationProblem` from Unit 1
bounds and initial points. It maps the extended specialist options into
`OptimisationOptions` and calls `run_multistart_minimisation` exactly once.
Unknown methods/overrides fail before callback execution. Iteration and
evaluation limits are forwarded without an internal retry loop.

### 12. Re-evaluate, filter, and order candidates

The coordinator forms an exact-coordinate union of backend candidates and Unit
1 deterministic initial points. Memoized evaluations are reused. Only complete
feasible evaluations are retained. The existing optimizer already clusters
backend minima; Unit 2 does not tolerance-quantize additional coordinates.

Candidates are ordered by physical aggregate objective and then exact decoded
coordinate tuple. `candidate_limit` counts the best candidate plus
alternatives, so alternatives contain at most `candidate_limit - 1` entries.
The parent process performs one canonical full re-evaluation per retained exact
coordinate before conversion to public results, independent of worker-local
memo state.
If no feasible candidate remains, a typed exhaustion error reports the best
available diagnostics rather than returning a least-infeasible success.

### 13. Assemble the result

The best and alternative candidates contain only Unit 1 frozen contracts and
finite canonical quantities. Termination metadata is translated from the
backend without retaining a backend object. Result diagnostics may contain
non-fatal numerical normalization notices; ordinary rejected-candidate details
are summarized rather than copying an unbounded evaluation trace.

## Failure Flow

1. Optimizer method/option errors fail before context targeting.
2. Scope/period/base-target/snapshot errors raise a typed context error.
3. Empty all-period bounds retain Unit 1's typed model error.
4. Candidate vector/order failures become ordinary infeasible evaluations.
5. Candidate targeting or coverage failures become ordinary infeasible
   evaluations with period and side evidence.
6. Non-positive kelvin, non-finite entropy, or materially negative entropy are
   candidate infeasibility when coordinate-dependent; invariant calculation
   failures raise a typed objective error.
7. Turbine failures follow the recoverability split defined above.
8. Existing optimizer validation/exhaustion failures translate to typed
   placement errors with method, seed, limits, and diagnostics.
9. No feasible retained candidate raises; no partial result is returned.

## Testable Properties - PBT-01

| Component | Category | Property requirement |
|---|---|---|
| Snapshot/context builder | Invariant | Context extraction preserves requested period order/weights, finite canonical values, and source-zone state. |
| Snapshot reconstruction | Idempotence | Reconstructing and targeting twice from the same snapshot/candidate yields equivalent allocations without accumulated state. |
| Bound-envelope population | Invariant | Every required Unit 1 coordinate appears exactly once per period with the correct scope and period identity. |
| Candidate replay | Easy verification | Every feasible evaluation passes independent vector, per-period coverage, finite-value, and objective checks. |
| Period replay | Induction | Adding one feasible period extends results by one and changes the aggregate by exactly its weighted contribution; adding an infeasible period makes the candidate infeasible. |
| Coverage | Invariant | Accepted hot/cold allocations conserve their respective residual demands within the named tolerance for every period. |
| Zero-duty inventory | Invariant | Adding or retaining a feasible zero-duty level preserves total coverage and keeps all declared identities. |
| Balanced-composite kernel | Oracle | Sensible intervals agree with exact logarithmic entropy and isothermal intervals agree with the signed `Q/T` limit. |
| Heat-load scaling | Metamorphic | Scaling every balanced heat-load interval by a positive factor scales entropy generation by the same factor. |
| Entropy branch aggregation | Commutativity | Permuting independent branch contributions changes the canonical sum by no more than the declared floating tolerance. |
| Thermodynamic result | Invariant | Total balanced-composite entropy equals utility plus process terms; exergy equals ambient kelvin times total; the result is finite and non-negative. |
| Default-utility penalty | Invariant | Positive generated `HU`/`CU` duty is infeasible regardless of level ordering; zero duty is unpenalized. |
| Thermal purchase evaluator | Oracle | Result equals the explicit unit-normalized sum of duty times level price. |
| Cogeneration filter | Invariant | Ineligible, cold, and zero-duty levels never reach the turbine adapter; eligible positive-duty hot levels retain deterministic descending order. |
| Monetary evaluator | Oracle | Net cost equals thermal cost minus electricity credit and credit equals work times electricity price after unit conversion. |
| Period aggregation | Oracle | Aggregate equals the explicit raw weighted sum of period objectives. |
| Period aggregation | Commutativity | Permuting period pairs preserves the mathematical aggregate within tolerance while public output order remains canonical. |
| Penalty mapping | Invariant | Every feasible transformed scalar is below one and every infeasible scalar is at least one; feasible physical ordering is preserved. |
| Exact evaluation memo | Idempotence | Any repeated exact coordinate sequence within one process/session invokes physical replay once and returns an equal immutable compact evaluation each time. |
| Evaluation memo command model | Easy verification | For generated per-session lookup sequences, memo size equals the number of distinct normalized tuples up to the evaluation budget and results equal a no-cache reference. |
| Candidate normalization | Invariant | Returned candidates are feasible, exact-coordinate unique, bounded by the limit, and ordered by physical objective then coordinates. |
| Result assembly | Round-trip | Unit 2 result through Unit 1 JSON serialization preserves structure/order and floats within named tolerances. |
| Optimization coordinator | Oracle | On generated tiny bounded analytical cases, the best feasible objective is equivalent to the structured-grid reference within grid resolution/tolerance. |
| Optimization coordinator | Invariant | Fixed method, seed, context, and options produce equivalent ordered candidates within documented tolerances. |
| Failure translation | Invariant | Each failure class retains stable code plus applicable scope, objective, counts, method, seed, period, side, and coverage context. |

There is no recursive business structure beyond ordered period/level folds, so
structural induction applies only to adding periods and contribution terms.
General API idempotency is N/A because service execution is an analysis, not a
state mutation; idempotence applies specifically to snapshot replay and the
per-run memo. No operation is generally commutative where caller-visible order
is part of the contract.

Every property above must be carried into Code Generation planning. Explicit
examples remain mandatory for hand-calculable entropy, zero-duty coverage,
monetary/cogeneration, one failed period, turbine recoverability, penalty
separation, direct and Total Site adapters, and optimizer exhaustion.

## PBT Compliance at Functional Design

| Rule | Status | Rationale |
|---|---|---|
| PBT-01 | Compliant | Every Unit 2 algorithmic component has named property categories and exact requirements; non-applicable induction, idempotence, and commutativity scopes are explained. |
| PBT-02 through PBT-10 | N/A at this stage | Their enforcement begins at later stages, but the identified round-trip, invariant, oracle, generator, reproducibility, state-model, and complementary-example obligations are carried forward explicitly. |

There is no blocking PBT finding. Security and Resiliency extensions remain
disabled for this feature and are not enforced at this stage.
