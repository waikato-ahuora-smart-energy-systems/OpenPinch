# CoolProp HPR and MVR implementation audit

## Scope and outcome

Audited all modified runtime owners, public keyword routing, contracts,
multiperiod coordination, tests, generated tutorials and completion claims.
Preserved existing work. No commit, deployment or publication performed.

The earlier passing test suite did not establish all of the claimed guarantees.
The following defects were reproduced and corrected, not merely documented.

## Findings and corrections

1. **Public budget keywords were unusable.** The accessor accepted them but
   the downstream runtime-context validator rejected them. The real public
   methods now accept and exercise both controls.
2. **Budgets were only optimizer hints.** Warm starts, restarts and local
   polishing could exceed the evaluation allowance. The shared objective now
   enforces a hard search-evaluation limit. HPR restarts and polishing are
   serial so they share the exact cache, counter and fatal-error boundary.
   Generic optimization retains its existing defaults; `local_method=None`
   allows HPR to own polishing without a worker pool swallowing exceptions.
3. **Backend failure could discard visited successes.** The adapter retains
   the best sixteen viable visited points in addition to warm starts and
   returned backend candidates. Exhaustion cannot discard an evaluated viable
   warm start. Failures include search counts, bounded representative reasons
   and budget-exhaustion evidence, including multiperiod searches.
4. **Capability preflight over-rejected valid fluids.** An R134a seed at
   150 degC fails saturation lookup while another point at 60 degC works.
   Fluid identity/backend rejection now precedes even Carnot initialization.
   Representative-state support is recorded, but a variable seed is not treated
   as proof that the entire search space is impossible. A typed failure of
   optional Carnot screening no longer prevents a physical-cycle search.
5. **Multi-stage penalties could crash before normalization.** Cascade and
   parallel cycles added differently sized NumPy penalty arrays. They now
   reduce each stage before summing. A generated oracle test covers unequal
   lengths and empty arrays.
6. **Result records could reject or misdescribe solved cycles.** Inactive
   stages legitimately have zero duty. Overall zero-duty candidates are now
   classified as unsuccessful before final records; individual zero-duty
   loops remain representable. VC+MVR uses the actual VC stage count for fluid
   roles, mode-specific useful duty, and configured process-port counts.
7. **Fatal errors could masquerade as physical infeasibility.** Narrowed broad
   catches in fluid resolution and MVR required-state evaluation. Errors in
   accounting and final record construction propagate. Thermodynamic solve
   and physical stream-profile failures remain candidate-local.
8. **Detachment checks were incomplete.** Final translation now returns the
   deep copy, excludes models before nested serialization and rejects arbitrary
   copyable objects in permissive output fields. Known stream/value containers
   are checked through their established reporting schemas. Typed HPR and
   direct-MVR exceptions support copying and process transport.
9. **Direct-MVR validation and fallback evidence had holes.** Compression
   controls are validated before property calls; unavailable source states
   have contextual errors. Non-finite optional saturation results trigger
   named fallbacks. Fallback context is bounded and indices are strict integers.
   A real first-stage/failed-second-stage test proves component atomicity.
10. **Tutorial success could be a no-op.** Notebook 09 rejects `None` as a
    feasible result and requires its target record and map. Notebook 11 uses
    a separately loaded case to prove optimized VC+MVR, because its original
    process-MVR case may have no residual heating load. Both real examples run.
    Notebook 10 now catches only typed HPR targeting failures and describes a
    completed batch without assuming every period requires a nonzero HPR target.
11. **Penalty types and diagnostic counts were overly permissive.** Penalties
    reject numeric strings and complex numbers instead of coercing them;
    diagnostic counts validate before Pydantic coercion. Invalid warm starts
    are rejected before calling the expensive objective.

## Verification inventory

- `tests/application/test_coolprop_hpr_audit.py`: ten real direct/utility
  heat-pump, refrigeration and VC+MVR solves; three real multi-stage record
  checks; one invalid-MVR-fluid preflight/atomicity check. No mocked
  thermodynamics or optimizer. Count instrumentation observes real search.
- `tests/analysis/heat_pumps/test_hpr_audit_regressions.py`: hard budget and
  stateful cache reference-model properties, unequal-length penalty oracle,
  invalid seeds/fluids, numeric types, record roles, copy/pickle behavior,
  fatal propagation, second-stage atomicity and unknown-object rejection.
- Existing result, search, contract, direct-MVR, multiperiod, generic optimizer,
  public API and notebook suites are rerun alongside these additions.
- Notebooks 09 and 11 execute all code cells in fresh namespaces. Notebook 09
  produces a three-point CoolProp map; notebook 11 produces direct process-MVR
  stages and a nonempty optimized VC+MVR record.
- A real three-period CoolProp batch (`turndown`, `base`, `peak`) completes with
  two workers, 80 evaluations per search and detached JSON result snapshots.
- Strict Sphinx, wheel/source build, installed-wheel TESPy smoke, Ruff,
  compilation and notebook regeneration are checked.

Final test counts are recorded in the audit addendum of `build-and-test-summary.md`.

## Exact limits and interpretation

- `maximum_evaluations` counts distinct objective evaluations within one search,
  including its warm starts, all restarts and polishing. Cache hits are free.
  Final artifact reevaluations are additional and counted in failure summaries.
  Optional Carnot initialization is a separate bounded search. A shared-design
  objective may evaluate multiple periods per call; a call is not a PropsSI
  invocation. This is not a wall-clock cancellation guarantee.
- `maximum_iterations` is an optimizer/polishing iteration control per run.
  It is not a universal count of thermodynamic property calls.
- Search caches are bounded by the evaluation allowance; memory therefore
  scales with the configured allowance and design-vector dimension. HPR now
  trades process-parallel throughput for a deterministic shared evaluation cap.
- State-probe failure is candidate-local when temperatures are optimization
  variables. Identity/backend failure is a global preflight rejection. This
  refines requirements FR6/VR2 to avoid invalid global conclusions from one seed.
- No global optimum, feasibility for every fluid, or minimum delivered duty
  beyond positivity is claimed. Existing economics can favor a very small
  positive heat-pump duty; changing that objective is outside this reliability
  audit. Multi-loop records do not make scalar-only performance-map adapters
  support multi-loop simulation.
- Public result snapshots and simulation records are JSON export surfaces.
  Internal domain target models also contain Configuration and stream objects;
  arbitrary direct JSON dumping of those domain models is not an export API.
- No live engine or figure is retained by the cache or public HPR output.
  Simulated search still builds the physical stream profiles required for
  cascade/economic evaluation; it omits final records and debug artifacts.

## Extension compliance

- PBT-01: compliant; design properties remain applicable, supplemented here by
  hard budget, command-sequence cache and unequal-stage penalty properties.
- PBT-02: compliant; budget/diagnostic JSON round trips remain covered.
- PBT-03: compliant; limits, finite penalties, ordering and record invariants.
- PBT-04: compliant; repeated-coordinate caching and penalty normalization.
- PBT-05: compliant; scalar-sum and bounded dictionary reference models.
- PBT-06: compliant; generated cache command sequences check model equivalence
  after each command, including empty sequences and exhausted budgets.
- PBT-07: compliant; bounded coordinates, evaluation budgets and stage vectors.
- PBT-08: compliant; fixed seed 20260924, normal shrinking, ordinary pytest
  discovery and no new property-test exclusions.
- PBT-09: compliant; existing Hypothesis/pytest dependencies reused.
- PBT-10: compliant; properties complement explicit real public regressions.
- Security and Resiliency extensions: disabled; skipped as configured.

This evidence supersedes the earlier blanket claim that every U1-P1 through
U3-P12 obligation had already been conclusively verified. Tests establish the
listed behaviors, not universal thermodynamic correctness.
