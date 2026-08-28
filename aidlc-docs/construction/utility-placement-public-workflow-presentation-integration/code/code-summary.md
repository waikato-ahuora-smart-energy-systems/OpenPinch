# Unit 3 Code Summary: Public Workflow and Presentation Integration

## Outcome

Unit 3 exposes utility placement through the existing problem and workspace
target namespaces, keeps numerical ownership in Unit 2, provides result-only
presentation, and ships exactly one executable notebook. There is no
utility-placement CLI and no package-root export beyond `PinchProblem` and
`PinchWorkspace`.

## Production Ownership

- `OpenPinch/application/utility_placement.py` builds detached direct or Total
  Site period contexts from isolated problem copies and delegates once to the
  Unit 2 service.
- `OpenPinch/application/_problem/accessors/target.py` owns the keyword-only
  problem and shared all-period target methods.
- `OpenPinch/application/problem.py` owns the detached successful-result cache
  and explicit/cached metrics, frame, and report methods.
- `OpenPinch/application/workspace.py` mirrors the target through the existing
  ordered `CaseBatchResult` behavior without changing the active case.
- `OpenPinch/presentation/utility_placement.py` performs pure result-to-view
  transformations and never starts target analysis or optimization.
- `scripts/generate_tutorial_notebooks.py` owns
  `19_utility_placement_optimisation.ipynb`, including the executable
  thermodynamic example plus the public named-case utility-replacement and
  standard plotting workflow.
- `OpenPinch/resources.py` and `docs/_data/tutorial-coverage.csv` own packaged
  discovery metadata and operation-level tutorial traceability.

## TDD Evidence

RED-GREEN slices covered isolated context extraction, public problem and
all-period invocation, invalid-before-analysis behavior, successful-result
caching, failure preservation, ordered workspace batches, pure presentation,
the exact-one generated notebook contract, documentation, input-change cache
invalidation, removal of the monetary public surface, physical
process-stream entropy extraction, exact Total Site residual calibration,
mixed-level named-case replacement, standard GCC/TSP plotting, and
installed-wheel graph display. The rejected placement-specific plot accessor
is explicitly absent.

The correction-focused utility-placement gate completed with 179 passing
tests. This includes all three utility-placement units, property and oracle
tests, performance thresholds, application/presentation integration, and the
new named-case direct/Total Site property. Separate notebook, documentation,
closed API inventory, packaging, and release gates pass.
The final case-based cross-owner gate passed 297 tests with 3 optional-profile skips and
adds a byte-stability check that prevents the generator from rewriting any
current notebook for Python formatting or environment-owned metadata alone.
All repository Ruff checks pass. Unit 3-owned application and presentation
modules have 97 percent branch coverage. Unit 2's separately scoped gate has
96 percent branch coverage.

## Property-Based Testing Compliance

| Rule | Status | Evidence |
| --- | --- | --- |
| PBT-01 | Compliant | Functional Design records the complete property inventory. |
| PBT-02 | Compliant | Request, vector, and public result JSON round trips are tested. |
| PBT-03 | Compliant | Bounds, ordering, coverage, objective, and source-state invariants are generated. |
| PBT-04 | Compliant | Fixed-seed repeat calls are equal and observation is side-effect free. |
| PBT-05 | Compliant | Decimal analytical and structured-grid oracles cover Unit 2. |
| PBT-06 | Compliant | Generated case order, failure isolation, active-case state, and cache behavior are tested. |
| PBT-07 | Compliant | Constrained strategies generate valid templates, periods, contexts, and cases. |
| PBT-08 | Compliant | Hypothesis shrinking and repository seed `20260715` are retained. |
| PBT-09 | Compliant | Existing Hypothesis and pytest dependencies are used. |
| PBT-10 | Compliant | Exact examples complement the property and oracle suites. |

Security and Resiliency extensions remain disabled. No applicable enabled
extension finding is unresolved.

## Compatibility and State Guarantees

- Existing target methods, `TargetOutput`, legacy result caches, source JSON,
  selected periods, and workspace active-case behavior remain intact.
- Placement analysis reconstructs its base target from isolated problem copies.
- Process entropy is extracted from period-resolved physical streams and
  explicit segments at real temperatures; rounded problem-table profiles are
  calibrated to the target's exact residual duty before strict coverage checks.
- Success updates only the dedicated immutable placement-result cache; typed
  failure leaves the previous successful result available.
- Loading or replacing inputs, updating options, or changing the temperature
  approach contribution invalidates the detached placement cache.
- Root exports and mandatory dependencies are unchanged.
- Results and nested alternatives contain JSON-safe frozen contracts rather
  than backend, callable, or mutable source objects.

## Notebook and Documentation

`OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb` is a
generator-owned base-profile notebook. It calls the public target surface once
with `isothermal=2` and `sensible=2` for physical balanced-composite entropy
generation. The call directly returns a normal case with eight optimized
utilities; `workspace.add(...)` registers it as `optimized_utilities`. The
notebook then runs ordinary direct and Total Site targets, displays a normal
summary, and uses the existing standard GCC and TSP methods. Its installed-
wheel smoke produced the retained entropy evidence and both figures. The user
guide documents concise counts, the thermodynamic objective, direct and
Total Site scope, periods, batches, units, errors, result interpretation, and
the absence of a new CLI.

## Distribution and Regression Evidence

- Complete case-based all-test run: 2,389 passed and 4 expected tests skipped; solver
  markers were enabled.
- Sphinx built 54 source pages with warnings treated as errors.
- No-isolation source and wheel builds completed without dependency download.
- Both archives contain all specialist, application, contract, presentation,
  and notebook assets.
- An isolated installed-wheel smoke imported the specialist and executed the
  thermodynamic notebook successfully.

## Known Limitations

- Optimization is bounded and deterministic for fixed inputs and seed but is
  not a proof of the global optimum.
- Monetary placement remains deferred pending a separate approved
  boiler/turbine requirements and design cycle.
- Result caching is process-local and intentionally has no persistence or
  distributed coordination.
- Utility placement recommends temperatures and duties; it does not mutate the
  source problem's utility definitions or synthesize an exchanger network.
  The notebook deliberately copies those results into a separate named case.
- The default objective evaluates physical entropy generation over
  piecewise-linear balanced composites; bounded stochastic search still does
  not prove global optimality.
