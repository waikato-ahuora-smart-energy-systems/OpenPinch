# Requirement-to-Unit Map

| Requirement group | Unit |
|---|---|
| FR-01 through FR-10, FR-16, FR-17 | Domain and Input |
| FR-11, FR-12 | Targeting and Integration |
| FR-13 through FR-15 | Heat Exchanger Network |
| NFR-01 through NFR-04, NFR-07 | Domain and Input |
| NFR-02, NFR-06 | Targeting and Integration |
| NFR-05 through NFR-07 | Heat Exchanger Network |

## Package Usability Refactor Story and Requirement Map

| Story or requirement | Primary unit | Supporting units |
|---|---|---|
| US-1 First solve | 2 | 1, 4, 5 |
| US-2 Total Site workflow | 2 | 1, 4, 5 |
| US-3 Scenario comparison | 3 | 1, 4, 5 |
| US-4 Advanced thermal workflows | 2 | 1, 3, 4, 5 |
| US-5 HEN selection and visualization | 3 | 1, 4, 5 |
| US-8 Predictable interaction | 2 | 1, 3, 4, 5 |
| FR-1 Canonical package boundary | 2 | 1, 4, 5 |
| FR-2 Explicit targeting vocabulary | 2 | 1, 4, 5 |
| FR-3 Friendly workflow configuration | 2 | 1, 3, 4, 5 |
| FR-4 Intuitive workspace scenarios | 3 | 1, 4, 5 |
| FR-5 Public result operations | 2 | 1, 3, 4, 5 |
| FR-6 Public HEN design experience | 3 | 1, 4, 5 |
| FR-7 Tutorial redesign | 4 | 1, 2, 3, 5 |
| FR-8 Executable tutorial contracts | 5 | 1, 4 |
| FR-9 Complete PinchProblem interaction contract | 2 | 1, 3, 4, 5 |
| Acceptance 1 notebook execution | 4 | 5 |
| Acceptance 2 public import boundary | 4 | 5 |
| Acceptance 3 retired syntax and private-helper removal | 4 | 5 |
| Acceptance 4 weighted HPR aggregation | 1 | 2, 4, 5 |
| Acceptance 5 root-only quickstart | 4 | 2, 5 |
| Acceptance 6 lint/docs/architecture/non-solver gates | 5 | 1, 2, 3, 4 |
| Acceptance 7 selector removal | 2 | 1, 3, 5 |
| Acceptance 8 complete problem interaction classification | 1 | 2, 3, 5 |
| Acceptance 9 tutorial manifest completeness | 4 | 1, 5 |
| Acceptance 10 multiperiod study templates | 4 | 2, 3, 5 |
| Acceptance 11 signature and stale-symbol guards | 5 | 1, 2, 3 |
| Acceptance 11 multi-segment example | 4 | 2, 5 |
| Acceptance 12 live inventory and RTD coverage parity | 5 | 1, 4 |
| Acceptance 13 retiring workspace methods absent | 3 | 1, 5 |
| Acceptance 14 100 percent executable coverage | 4 | 5 |
| Acceptance 15 warning-free generated RTD coverage | 5 | 4 |

Every story, FR section, and numbered acceptance criterion is assigned. Unit 1
owns contract evidence, Units 2 and 3 own runtime behavior, Unit 4 owns teaching
coverage, and Unit 5 owns public documentation and enforcement. There are no
orphan stories and no story is assigned only to a downstream consumer without
an implementation owner.

## Repository Issue Remediation Requirement Map

User Stories are intentionally skipped for these bounded correctness fixes.
Functional requirements, NFRs, and acceptance criteria provide full assignment
traceability instead.

| Requirement | Primary unit | Supporting unit |
|---|---|---|
| FR-1 Workspace identifiers and export containment | 1 | 3 |
| FR-2 Detached problem-input observation | 1 | 3 |
| FR-3 Exact OpenHENS checkout identity | 2 | 3 |
| FR-4 Collision-free workbook allocation | 1 | 3 |
| FR-5 Consistent unloaded-problem error | 1 | 3 |
| FR-6 Current contract documentation | 3 | 1 and 2 |
| NFR Safety and filesystem containment | 1 | 3 |
| NFR Reliability and concurrent allocation | 1 | 3 |
| NFR State consistency | 1 | 3 |
| NFR Deterministic checkout identity | 2 | 3 |
| NFR Cross-platform portability | 1 | 3 |
| NFR Maintainability and shared validation | 1 | 3 |
| NFR No new runtime dependencies | 1 and 2 | 3 |
| NFR Bounded performance overhead | 1 | 3 |

| Acceptance criterion | Primary unit | Final verification |
|---|---|---|
| 1. Six reproductions have focused regressions | 1 and 2 | 3 |
| 2. Unsafe identifiers rejected and exports contained | 1 | 3 |
| 3. Snapshot mutation cannot affect internal state | 1 | 3 |
| 4. OpenHENS modules originate from requested checkout | 2 | 3 |
| 5. Repeated/concurrent exports never collide | 1 | 3 |
| 6. Empty multiplier update raises canonical error | 1 | 3 |
| 7. Current docs contain no retired API claims | 3 | 3 |
| 8. Full quality and distribution gates pass | 3 | 3 |

Every FR, NFR, and numbered acceptance criterion is assigned to an implementation
owner and a final verification owner. Unit 3 cannot redefine Unit 1 or Unit 2
behavior; it documents and enforces their completed contracts.

## Utility Placement Optimisation Story and Requirement Map

### User stories

| Story | Primary unit | Supporting unit(s) | Delivery evidence |
|---|---|---|---|
| UPO-01 Define utility-level families | 1 | 2 | Count/template validation, vector schema, mixed-level properties |
| UPO-02 Run the default placement journey | 3 | 1 and 2 | Public default-thermodynamic direct/Total Site workflows and executable notebook |
| UPO-03 Cover all heating and cooling demand | 2 | 1 | Per-period hot/cold conservation examples and properties |
| UPO-04 Minimise balanced-composite entropy generation | 2 | 1 | Analytical logarithmic/isothermal cases and closer-temperature ranking evidence |
| UPO-05 Evaluate monetary cost and cogeneration | 2 | 1 and 3 | Cost decomposition, eligible-level turbine fixtures, and executable notebook workflow |
| UPO-06 Select one multiperiod placement | 2 | 3 | Shared-vector replay, weights, and one-failed-period tests |
| UPO-07 Inspect and compare candidates | 2 | 1 and 3 | Ordered alternatives plus presentation integration |
| UPO-08 Recover from invalid/infeasible studies | 2 | 1 and 3 | Typed validation, candidate, solve, and public failure tests |
| UPO-09 Integrate a stable detached contract | 1 | 2 and 3 | JSON, specialist import, copy, and no-mutation tests |
| UPO-10 Run ordered case batches | 3 | 1 and 2 | Batch ordering, identity, isolation, and state properties |
| UPO-11 Verify deterministic bounded optimisation | 2 | 1 | Fixed seed, budget, stable ordering, and grid-oracle evidence |
| UPO-12 Deliver through TDD and full property testing | 1, 2, and 3 | None | Per-unit red-green-refactor and PBT compliance records |

Every story has one primary implementation owner except UPO-12, whose purpose
is explicitly cross-cutting. Supporting ownership identifies consumed contracts
or final integration, not duplicated business logic.

### Unit 1 delivery status

- [x] UPO-01 Unit 1 primary slice: count/template validation, deterministic
  blueprints, mixed isothermal/sensible vector schema, bounds, and starts.
- [x] UPO-08 Unit 1 supporting slice: typed request/template/model failures and
  structured ordinary candidate diagnostics.
- [x] UPO-09 Unit 1 primary slice: frozen detached schemas, nested JSON
  round-trips, specialist imports, and protected root API.
- [x] UPO-12 Unit 1 slice: recorded RED-GREEN-REFACTOR evidence, complete
  property testing, fixed-seed reproduction, shrinking, coverage, performance,
  regression, and packaging gates.

UPO-08 remains open at workflow level until Unit 2 service failures and Unit 3
public recovery are integrated. UPO-12 remains cross-cutting until all three
units complete their TDD and property-testing obligations.

### Integrated delivery status

- [x] UPO-01 through UPO-04: level contracts, complete allocation, and the
  thermodynamic objective are implemented and verified.
- [x] UPO-05: monetary purchase cost, turbine work, electricity credit, and net
  objective are implemented and demonstrated with positive cogeneration.
- [x] UPO-06 and UPO-07: shared-period placement, ordered alternatives, and
  result-only presentation are implemented and verified.
- [x] UPO-08 and UPO-09: typed recovery, detached immutable results, JSON
  round-trips, and success/failure source-state preservation are verified.
- [x] UPO-10: generated workspace case ordering, failure isolation, and
  active-case preservation are verified.
- [x] UPO-11: fixed-seed bounded optimization, analytical checks, decimal and
  structured-grid oracles, performance gates, and real-backend regression are
  verified.
- [x] UPO-12: every unit completed approved RED-GREEN-REFACTOR, applicable
  PBT-01 through PBT-10 evidence, regression, documentation, and distribution
  gates.

All UPO stories are complete at integrated Build and Test. Delivery evidence is
summarized in
`aidlc-docs/construction/build-and-test/build-and-test-summary.md`.

### Functional requirements

| Requirement | Primary unit | Supporting/final verification |
|---|---|---|
| FR-001 Public workflow | 3 | Units 1 and 2 provide contract/service |
| FR-002 Objective selection | 1 | Units 2 and 3 |
| FR-003 Level-count validation | 1 | Unit 3 pre-execution integration test |
| FR-004 Template validation | 1 | Unit 2 model consumption |
| FR-005 Bounds and starting candidates | 1 | Unit 2 period-context bounds |
| FR-006 Ordering and separation | 1 | Unit 2 feasibility/ranking |
| FR-007 Direct and indirect scope | 2 | Unit 3 scope resolver/API |
| FR-008 Multiperiod feasibility | 2 | Unit 3 period selection |
| FR-009 Thermodynamic evaluation | 2 | Unit 1 breakdown contracts |
| FR-010 Monetary evaluation and cogeneration | 2 | Unit 1 price/eligibility contracts |
| FR-011 Deterministic optimisation | 2 | Unit 1 options/order contracts |
| FR-012 Constraint handling | 2 | Unit 1 error schema; Unit 3 public translation |
| FR-013 Detached result | 3 | Units 1 and 2 detached values |
| FR-014 Typed result and alternatives | 1 and 2 | Unit 3 public serialization |
| FR-015 Reporting integration | 3 | Units 1 and 2 result contract/content |
| FR-016 Batch isolation | 3 | Unit 2 typed failures |
| FR-017 Executable notebook example | 3 | Units 1 and 2 provide stable public contract and service |

### Non-functional requirements

| Requirement | Primary unit(s) | Final verification |
|---|---|---|
| NFR-001 Numerical correctness | 1 and 2 | Unit 3 end-to-end regressions |
| NFR-002 Reproducibility | 1 and 2 | Unit 3 public and batch repeats |
| NFR-003 Bounded execution | 2 | Unit 3 option forwarding |
| NFR-004 Compatibility | 1 and 3 | Integrated package/distribution gate |
| NFR-005 Maintainability | 1, 2, and 3 | Architecture and import-direction tests |
| NFR-006 Observability | 1 and 2 | Unit 3 public and batch diagnostics |
| NFR-007 Security/resiliency scope | 3 | Verify no new boundary or dependency |

### Property-Based Testing extension ownership

| Rule | Unit ownership | Applicability and planned evidence |
|---|---|---|
| PBT-01 Property identification | 1, 2, and 3 | Blocking per-unit Functional Design property inventory |
| PBT-02 Round trips | 1 primary; 3 integration | Vector and JSON round-trips, including public result serialization |
| PBT-03 Invariants | 1 and 2 primary; 3 integration | Bounds, ordering, spans, coverage, feasibility, objectives, and state |
| PBT-04 Idempotency | N/A unless Functional Design identifies a stateful repeat operation | Pure evaluators use equality/repeatability; cache writes require no duplicated effect |
| PBT-05 Oracle/model based | 2 | Analytical equations and bounded structured-grid placement oracle |
| PBT-06 Stateful testing | 3 | Workspace batch ordering, active-case preservation, cache/source non-mutation |
| PBT-07 Generator quality | 1 primary; 2 and 3 consumers | Domain-valid templates, contexts, periods, weights, and small feasible cascades |
| PBT-08 Shrinking/reproducibility | 1, 2, and 3 | Fixed CI seed, shrinking enabled, minimal regressions retained |
| PBT-09 Framework selection | 1, 2, and 3 | Existing Hypothesis and pytest; confirm during each NFR Requirements stage |
| PBT-10 Complementary strategy | 1, 2, and 3 | Example tests for exact cases plus properties for generated domains |

No rule is treated as implemented at Units Generation. These assignments become
blocking at the stages named by the enabled extension's applicability matrix.

### Example-based acceptance ownership

| Acceptance requirement | Primary unit | Integrated verification |
|---|---|---|
| 1. Invalid counts and template mismatch | 1 | 3 |
| 2. Mixed-level encode/decode | 1 | 2 |
| 3. Target temperature derivation | 1 | 2 |
| 4. Hot/cold sensible and near-isothermal entropy | 2 | 3 |
| 5. Lower-entropy two-level ranking | 2 | 3 |
| 6. Thermal cost minus cogeneration credit | 2 | 3 |
| 7. Shared placement replay and weighted sum | 2 | 3 |
| 8. Complete hot/cold coverage and zero-duty level | 2 | 3 |
| 9. One-period coverage failure cannot win | 2 | 3 |
| 10. Direct and Total Site samples | 2 | 3 |
| 11. Source problem and case non-mutation | 3 | 3 |
| 12. Result JSON round-trip | 1 | 3 |
| 13. Fixed-seed ordered repeat | 2 | 3 |
| 14. Typed diagnostics across failure classes | 1 and 2 | 3 |
| 15. Workspace batch ordering and isolation | 3 | 3 |
| 16. Executable two-objective notebook and package registration | 3 | 3 |

### TDD and quality-gate ownership

- Every unit begins Code Generation only after approved Functional Design, NFR
  Requirements, NFR Design, and a checkbox-based code-generation plan.
- Every production slice starts with a focused failing example or property test,
  adds the minimum production behavior, and refactors only while focused tests
  stay green.
- Unit 1 owns reusable strategies and contract fixtures; Unit 2 owns analytical
  and oracle fixtures; Unit 3 owns public, batch, packaging, and docs fixtures.
- Unit-level gates run before dependent work. Unit 3 and Build and Test own the
  complete non-solver regression, fixed-seed PBT, Ruff, architecture, docs,
  wheel/source build, isolated installation, and import smoke gates.

### Coverage validation

Validation set: UPO-01, UPO-02, UPO-03, UPO-04, UPO-05, UPO-06, UPO-07,
UPO-08, UPO-09, UPO-10, UPO-11, UPO-12; FR-001, FR-002, FR-003, FR-004,
FR-005, FR-006, FR-007, FR-008, FR-009, FR-010, FR-011, FR-012, FR-013,
FR-014, FR-015, FR-016, FR-017; NFR-001, NFR-002, NFR-003, NFR-004, NFR-005,
NFR-006, NFR-007; PBT-01, PBT-02, PBT-03, PBT-04, PBT-05, PBT-06, PBT-07,
PBT-08, PBT-09, and PBT-10.

Every approved story, functional requirement, non-functional requirement,
example-based acceptance item, TDD obligation, and enabled PBT rule has an
implementation or verification owner. Security and Resiliency extensions
remain disabled and introduce no unit work.
