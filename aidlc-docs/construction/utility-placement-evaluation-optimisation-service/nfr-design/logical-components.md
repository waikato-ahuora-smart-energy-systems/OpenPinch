# Unit 2 Logical Components

## Production Component Map

| Logical component | Proposed owner | Primary input/output | Complexity/bound |
|---|---|---|---|
| Options reconciliation | contracts plus `optimisation.py` | Specialist options to existing optimizer options | Linear in override count |
| Context builder | `context.py` | Isolated source to frozen placement context/envelope | `O(P*(I+D))` plus existing target work |
| Snapshot reconstructor | `allocation.py` | Frozen period snapshot to fresh local target inputs | `O(I)` |
| Utility allocation adapter | `allocation.py` | Decoded levels/context to period allocation slices | `O(L+I)` plus existing cascade |
| Coverage verifier | `allocation.py` | Allocation and residual demand to status/diagnostic | `O(L)` |
| Entropy evaluator | `thermodynamics.py` | Allocation slices to thermodynamic breakdown | `O(L+I)` |
| Monetary evaluator | `economics.py` | Level duties/prices/work to monetary breakdown | `O(L)` |
| Cogeneration adapter | `cogeneration.py` | Eligible hot levels/settings to detached work | `O(L)` plus existing turbine |
| Penalty mapper | `penalties.py` | Physical objective or violation to backend scalar | Constant |
| Worker evaluation session | `evaluation.py` | Coordinate to compact evaluation | One `O(P*(L+I))` cold replay |
| Process-local memo | `evaluation.py` | Exact coordinate to compact record | Average `O(1)`, at most evaluation limit |
| Diagnostic accumulator | `evaluation.py` | Rejected diagnostics to counts/top ten | Constant/bounded per rejection |
| Optimization coordinator | `optimisation.py` | Model/session/options to candidate points | Existing backend plus `O(K log K)` normalization |
| Parent candidate evaluator | `optimisation.py` | Deduplicated points to full evaluations | Bounded by starts/backend candidates |
| Service/result assembler | `service.py` | Request/context to frozen result | Sum of delegated stages |
| Operational errors | `errors.py` | Stable code/context/cause to typed exception | Constant per error |

Exact filenames may be combined during Code Generation when cohesion improves,
but ownership, dependency direction, and performance boundaries remain fixed.

## Data and Control Flow

1. Unit 3 later supplies an isolated resolved source to the context builder.
2. The context builder extracts frozen snapshots and a Unit 1 feasibility
   envelope, then Unit 1 constructs the placement model.
3. The coordinator maps validated options and passes a pickle-safe worker
   evaluation payload to the existing optimizer service.
4. Each process lazily creates a local memo. A cold coordinate is decoded,
   replayed across periods, checked for coverage, and evaluated; a repeated
   exact coordinate uses its compact record.
5. Feasible physical objectives and infeasible violations are mapped to
   disjoint backend scalar ranges.
6. The parent unions/deduplicates returned points and deterministic starts,
   then performs canonical full re-evaluation.
7. Feasible candidates are physically ordered, bounded, and converted into
   frozen Unit 1 result contracts; internal sessions and snapshots are dropped.

This numbered flow is the text representation; no infrastructure or remote
message flow exists.

## Component Contracts

### Context builder

- Accepts only an isolated source and resolved direct/Total Site identity.
- Produces immutable canonical period values and exact blueprint-coordinate
  bounds.
- Rejects `auto`, period mismatch, incomplete profiles, invalid ambient values,
  or mutable/runtime leakage.
- Leaves the source observably unchanged on success and failure.

### Allocation and coverage

- Reconstruct fresh existing owner inputs for every candidate/period.
- Preserve template keys and actual interval assignment.
- Calculate raw hot/cold residuals and use the central combined criterion.
- Retain zero-duty levels and never repair, rescale, or clip a failure.

### Thermodynamic evaluator

- Use real kelvin, stable sensible/limit formulas, and canonical finite sums.
- Return utility, process, total entropy, ambient, and exergy values.
- Provide branch-level internal evidence for test verification without exposing
  an unbounded public trace.

### Monetary and cogeneration evaluators

- Calculate purchase cost for every allocated level.
- Forward only eligible positive-duty hot levels to one fresh turbine.
- Resolve settings by explicit-over-configuration precedence.
- Separate thermal cost, work, electricity price/credit, and net cost.
- Classify correctable and run-level failures without parsing backend prose.

### Worker session and memo

- Serialize frozen evaluation inputs, excluding lock/memo state.
- Lazily recreate a lock-protected process-local memo after unpickling.
- Store compact bounded records keyed by exact normalized coordinates.
- Produce one finite scalar and keep physical/diagnostic data separate.

### Optimization coordinator and parent evaluator

- Call the existing public optimizer once with exact bounds/starts/options.
- Union backend points and starts, exact-deduplicate, and canonically replay in
  the parent.
- Retain only proven feasible candidates, order by physical objective and
  coordinates, and enforce the total candidate limit.
- Translate validation/operational/exhaustion failures to specialist errors.

### Service and result assembler

- Coordinate existing owners without duplicating their equations.
- Return one complete detached result or raise one typed error.
- Publish physical quantities, units, reproducibility metadata, coverage, and
  bounded diagnostics; omit transformed scalars, memos, workers, and backends.

## Failure Routing

| Failure source | Component response | Recovery |
|---|---|---|
| Invalid method/options | Options mapper raises typed validation error before targeting | Caller correction |
| Invalid scope/period/snapshot | Context error | Caller/application correction |
| Empty physical bounds | Existing Unit 1 model error | Caller/template correction |
| Candidate vector/order | Worker compact infeasible record | Continue bounded search |
| Candidate target/coverage | Worker compact infeasible record with period/side | Continue bounded search |
| Coordinate-dependent entropy/turbine | Worker compact infeasible record | Continue bounded search |
| Context-independent target/turbine defect | Typed run-level error with safe cause | None in Unit 2 |
| Backend error/exhaustion | Typed optimization/no-feasible error | Caller may issue new request |
| Public result invariant | Typed internal placement error; no partial result | Code correction |

## Verification Components

| Test component | Responsibility |
|---|---|
| Example suites | Pin every Functional Design scenario and failure boundary |
| Reusable strategies | Generate contexts, allocations, branches, economics, violations, memo commands, and tiny models |
| Analytical profile oracle | Verify zero identity, constant gap, symmetry, widening, and breakpoint refinement independently |
| Structured-grid oracle | Verify tiny bounded placement selection |
| Memo command model | Compare generated process-local lookup sequences with a no-cache reference |
| Multiprocess simulation | Verify pickle reconstruction, isolation, and canonical parent output |
| Real optimizer smoke | One tiny fixed-seed dual-annealing service regression |
| Performance suite | Enforce 50 ms, 1 second, and 1 ms p95 gates |
| Compatibility/architecture | Protect root API, dependency direction, existing owners, and options defaults |
| Packaging smoke | Import/execute specialist service from built artifacts |

## Infrastructure Components

| Component type | Status | Rationale |
|---|---|---|
| External queue/worker | N/A | Existing optimizer owns optional local processes |
| Shared/distributed cache | N/A | Process-local bounded memo is sufficient |
| Circuit breaker/retry controller | N/A | No remote dependency and retries are prohibited |
| Database/object storage | N/A | Service is detached and stateless |
| Network/API gateway | N/A | In-process Python service only |
| Auth/secret manager | N/A | No credential boundary; Security extension disabled |
| Monitoring service | N/A | Typed returned context; Unit 3 owns presentation |
| Deployment resource | N/A | Existing package distribution only |

## NFR Traceability

| NFR range | Components/patterns |
|---|---|
| U2-NFR-001 through U2-NFR-003 | Context/allocation/session/memo/diagnostic components; U2-NFRP-02 through U2-NFRP-04, U2-NFRP-11 through U2-NFRP-14 |
| U2-NFR-004 through U2-NFR-007 | Kernel/replay/memo components; U2-NFRP-16 |
| U2-NFR-008 through U2-NFR-013 | Coverage/thermodynamic/monetary/penalty components; U2-NFRP-05 through U2-NFRP-10 |
| U2-NFR-014 through U2-NFR-017 | Worker session/memo/parent coordinator; U2-NFRP-04, U2-NFRP-11 through U2-NFRP-13 |
| U2-NFR-018 through U2-NFR-025 | Error/diagnostic/service/result components; U2-NFRP-09, U2-NFRP-14, U2-NFRP-19 |
| U2-NFR-026 through U2-NFR-029 | Options/owners/architecture; U2-NFRP-01, U2-NFRP-15, U2-NFRP-18 |
| U2-NFR-030 through U2-NFR-032 | Verification components; U2-NFRP-07, U2-NFRP-16, U2-NFRP-17 |

All 32 Unit 2 NFRs have production or verification owners. No infrastructure
component is required.
