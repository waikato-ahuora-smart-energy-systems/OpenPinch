# Utility Placement User Stories

## Story Organization

Twelve stories use the approved hybrid structure. UPO-01 through UPO-08 form
the process-engineer journey. UPO-09 and UPO-10 cover public integration and
batch behavior. UPO-11 and UPO-12 are bounded numerical-assurance and
maintenance enablers. Every story uses Given/When/Then scenarios and direct
requirements/PBT traceability.

## Process-Engineer Journey

### UPO-01: Define Utility-Level Families

**Personas**: P-01 primary; P-02 supporting

As a process integration engineer, I want to define separate isothermal and
sensible utility-level counts and templates, so that the placement problem
represents the available utility families and plant constraints.

#### Scenario: Valid counts and templates

**Given** at least two isothermal levels and zero or more sensible levels per
side, **when** I submit matching hot and cold templates, **then** the request is
accepted with stable identities, units, bounds, price metadata, and
cogeneration eligibility. `[FR-003, FR-004; NFR-001]`

#### Scenario: Sensible and near-isothermal variables

**Given** sensible and isothermal templates, **when** the decision vector is
prepared, **then** sensible levels expose supply temperature and utility span
while isothermal levels retain the configured near-zero span.
`[FR-004, FR-005; NFR-001, NFR-005; PBT-02, PBT-03]`

#### Scenario: Invalid inventory

**Given** too few isothermal levels, negative or non-integral counts, duplicate
names, or mismatched templates, **when** validation runs, **then** it fails
before targeting or optimisation with a typed field-specific error.
`[FR-003, FR-004; NFR-006]`

### UPO-02: Run the Default Placement Journey

**Personas**: P-01 primary; P-02 supporting

As a process integration engineer, I want one public placement workflow with
thermodynamic cost as its default, so that I can obtain a feasible result
without learning private services or backend details.

#### Scenario: Direct scope

**Given** a validated process zone and utility templates, **when** I call
`problem.target.utility_placement(...)` without an objective override, **then**
the service runs direct placement with the thermodynamic objective and returns
a typed detached result. `[FR-001, FR-002, FR-007, FR-013, FR-014]`

#### Scenario: Total Site scope

**Given** a compatible site hierarchy, **when** I select the indirect or Total
Site base target, **then** the same workflow evaluates site utility profiles and
returns the same result shape with explicit scope metadata.
`[FR-001, FR-007, FR-014]`

#### Scenario: Scope incompatibility

**Given** an ambiguous or incompatible base target, **when** execution is
requested, **then** the service rejects it with guidance naming the supported
direct and Total Site choices. `[FR-007, FR-012; NFR-006]`

### UPO-03: Cover All Heating and Cooling Demand

**Personas**: P-01 primary; P-03 reviewer

As a process integration engineer, I want every successful candidate to cover
all residual heating and cooling demand, so that a low objective never hides an
unserved process requirement.

#### Scenario: Complete coverage

**Given** one candidate and one period, **when** existing cascades allocate
utility duties, **then** summed hot duties equal residual heating demand and
summed cold duties equal residual cooling demand within the named tolerance.
`[FR-008, FR-012; NFR-001; PBT-03]`

#### Scenario: Zero-duty level

**Given** more valid levels than the period needs, **when** one level receives
zero duty but both side balances close, **then** the candidate remains feasible
and preserves the requested level inventory. `[FR-004, FR-006, FR-008]`

#### Scenario: Unmet demand

**Given** a candidate with unmet or excess heating or cooling allocation,
**when** feasibility is checked, **then** it cannot rank as feasible and the
diagnostics quantify the failing side and duty. `[FR-008, FR-012; NFR-006]`

### UPO-04: Minimise Balanced-Composite Entropy Generation

**Personas**: P-01 primary; P-03 reviewer

As a process integration engineer, I want placement ranked by physical entropy
generation from the balanced process-plus-utility composites by default, so
that selected utility levels follow the process demand profile with lower
thermodynamic irreversibility.

#### Scenario: Analytical entropy

**Given** hand-calculable sensible and isothermal balanced-composite intervals,
**when** thermodynamic cost is evaluated, **then** it equals the expected
`CP * ln(T_out / T_in)` and signed `Q / T` terms within tolerance.
`[FR-009; NFR-001, NFR-005; PBT-03, PBT-05]`

#### Scenario: Temperature approach

**Given** two balanced utility placements serving the same process heat-load
intervals, **when** their costs are calculated, **then** the placement with
utility temperature closer to the matched process temperature has the lower
finite entropy generation. `[FR-004, FR-009; NFR-001; PBT-03, PBT-05]`

#### Scenario: Thermodynamic ranking

**Given** two feasible placements with identical demand coverage, **when**
their thermodynamic objectives differ, **then** the smaller balanced-composite
entropy-generation candidate ranks first and reports exergy as ambient
absolute temperature times entropy generation. `[FR-002, FR-009, FR-014; PBT-03]`

### UPO-05: Evaluate Monetary Cost and Cogeneration

**Personas**: P-01 primary; P-02 integrator; P-03 reviewer

As a process integration engineer, I want to minimise purchased thermal cost
after eligible cogeneration credit, so that utility placement reflects both
heat purchases and recoverable power value.

#### Scenario: Net monetary objective

**Given** valid template prices, an electricity export price, and eligible hot
levels, **when** I select the monetary objective, **then** the result separately
reports thermal purchase cost, cogenerated work, electricity credit, and their
net objective. `[FR-002, FR-010, FR-014; NFR-001]`

#### Scenario: Cogeneration eligibility

**Given** a mixture of eligible and ineligible hot templates, **when** turbine
work is evaluated, **then** only explicitly eligible levels enter the existing
multi-stage steam-turbine service. `[FR-004, FR-010; NFR-005]`

#### Scenario: Missing economics

**Given** missing or dimensionally invalid monetary inputs, **when** monetary
placement is requested, **then** validation fails before optimisation and names
the missing price or unit. `[FR-003, FR-010, FR-012; NFR-006]`

#### Scenario: Executable two-objective notebook

**Given** the generated utility-placement tutorial notebook, **when** it is
executed from top to bottom under its declared dependency profile, **then** it
uses only public Python APIs to run the default thermodynamic workflow and a
monetary workflow with eligible cogeneration, inspects both result
decompositions, and completes its lightweight assertions without a CLI step.
`[FR-001, FR-002, FR-010, FR-014, FR-017; NFR-004]`

### UPO-06: Select One Multiperiod Placement

**Personas**: P-01 primary; P-03 reviewer

As a process integration engineer, I want one placement feasible across every
operating period, so that the selected utility system works for the whole study
rather than an averaged surrogate.

#### Scenario: Per-period replay

**Given** ordered operating periods and one candidate vector, **when** it is
evaluated, **then** the same temperatures and template ordering are replayed
against every period independently. `[FR-008; NFR-001]`

#### Scenario: Weighted objective identity

**Given** non-negative period weights with at least one positive value, **when**
per-period costs are aggregated, **then** the aggregate equals the explicit raw
sum of weight times period cost. `[FR-008, FR-014; PBT-03]`

#### Scenario: One failed period

**Given** full demand coverage in most periods but a coverage or bound failure
in one period, **when** candidate feasibility is resolved, **then** the whole
candidate is infeasible and cannot rank first. `[FR-008, FR-012; NFR-006]`

### UPO-07: Inspect and Compare Placement Candidates

**Personas**: P-01 primary; P-02 integrator; P-03 reviewer

As a process integration engineer, I want the best placement and ordered
alternatives with decomposed evidence, so that I can review trade-offs rather
than act on an opaque scalar minimum.

#### Scenario: Best and alternatives

**Given** several feasible candidates, **when** the solve completes, **then**
the result exposes the best and bounded alternatives ordered by objective then
deterministic coordinate tie-breaker. `[FR-011, FR-014; NFR-002]`

#### Scenario: Engineering evidence

**Given** one returned candidate, **when** I inspect it, **then** I can see each
hot and cold level's kind, supply and target temperatures, duty, per-period
values, coverage, and objective decomposition with units. `[FR-014, FR-015]`

#### Scenario: Reporting comparison

**Given** solved placements in supported problem or workspace contexts,
**when** metrics, summaries, comparisons, or reports are requested, **then**
semantically applicable placement fields are available without hidden reruns.
`[FR-015; NFR-004]`

### UPO-08: Recover from Invalid or Infeasible Studies

**Personas**: P-01 primary; P-02 integrator; P-03 reviewer

As a process integration engineer, I want failures classified with actionable
diagnostics, so that I can correct inputs, bounds, prices, scope, or feasibility
instead of receiving a plausible but invalid result.

#### Scenario: Empty feasible region

**Given** template bounds, separation, and process feasibility have no common
region, **when** preprocessing completes, **then** a typed error identifies the
conflicting bounds before backend execution. `[FR-005, FR-006, FR-012; NFR-006]`

#### Scenario: Optimiser exhaustion

**Given** bounded execution that produces no feasible candidate, **when** the
evaluation budget ends, **then** a typed error includes scope, objective,
counts, periods, method, seed, and useful constraint diagnostics.
`[FR-011, FR-012; NFR-002, NFR-003, NFR-006]`

#### Scenario: No least-infeasible success

**Given** only penalised infeasible candidates, **when** results are normalized,
**then** none is exposed as a successful best placement. `[FR-012; PBT-03]`

## Integration Stories

### UPO-09: Integrate a Stable Detached Contract

**Personas**: P-02 primary; P-04 maintainer

As a Python library integrator, I want a typed, detached, serializable placement
contract, so that automated tools can persist and consume results without
private backend objects or source-study mutation.

#### Scenario: Public and specialist imports

**Given** an installed core package, **when** I use the feature, **then** the
workflow is available from `problem.target` and specialist contracts come from
their concrete modules without expanding package-root exports or mandatory
dependencies. `[FR-001; NFR-004]`

#### Scenario: Serialization round-trip

**Given** a solved result with nested candidates and periods, **when** it is
dumped to supported JSON and validated again, **then** public values and order
round-trip within declared tolerances and no callable or backend-private object
appears. `[FR-014; NFR-004; PBT-02, PBT-07]`

#### Scenario: No mutation

**Given** a loaded problem or workspace case, **when** placement succeeds or
fails, **then** process streams, utilities, configuration, targets, case
selection, and cached source data remain observably unchanged.
`[FR-013; NFR-005; PBT-03]`

### UPO-10: Run Ordered Case Batches

**Personas**: P-02 primary; P-01 supporting; P-04 maintainer

As a Python library integrator, I want workspace batches to run utility
placement in case order and isolate failures, so that scenario portfolios
remain automatable and comparable.

#### Scenario: Ordered success

**Given** several valid named cases, **when** placement runs through the case
batch surface, **then** results preserve requested case order and each result
retains its case and scope identity. `[FR-001, FR-016; NFR-002]`

#### Scenario: Failure isolation

**Given** valid and invalid cases in one batch, **when** the batch completes,
**then** successful placements remain available and each failed case retains a
typed isolated exception. `[FR-012, FR-016; NFR-006]`

#### Scenario: Batch non-mutation

**Given** an active workspace case, **when** a different ordered batch runs,
**then** the active case and all canonical case inputs remain unchanged.
`[FR-013, FR-016; PBT-06]`

## Assurance Enabler Stories

### UPO-11: Verify Deterministic Bounded Optimisation

**Personas**: P-03 primary; P-04 maintainer; P-01 beneficiary

As a numerical assurance reviewer, I want deterministic bounded optimisation
checked against transparent small-case oracles, so that candidate ranking is
reproducible and mathematically credible.

#### Scenario: Fixed-seed reproduction

**Given** identical inputs, environment, method, seed, and options, **when** the
solve is repeated, **then** ordered candidates, objectives, and termination
metadata are equivalent within documented tolerances. `[FR-011; NFR-002; PBT-08]`

#### Scenario: Bounded execution

**Given** explicit iteration and objective-evaluation limits, **when** the
backend runs, **then** it honours those limits without unbounded restart or
retry behavior. `[FR-011; NFR-003]`

#### Scenario: Reference oracle

**Given** a generated small analytical placement problem, **when** the solver's
best candidate is compared with a brute-force or structured-grid reference,
**then** feasibility and objective agree within the declared resolution and
tolerance. `[FR-005, FR-011; NFR-001; PBT-05, PBT-07]`

### UPO-12: Deliver Through TDD and Full Property Testing

**Personas**: P-04 primary; P-03 reviewer; all personas beneficiaries

As an OpenPinch maintainer, I want every production slice driven by focused
examples and complementary properties, so that the new analysis remains
correct, maintainable, reproducible, and compatible.

#### Scenario: Red-green-refactor evidence

**Given** one planned production slice, **when** it is implemented, **then** a
focused failing test exists first, the smallest implementation makes it pass,
and refactoring occurs only with focused tests green. `[NFR-005; TDD; PBT-10]`

#### Scenario: Domain properties and shrinking

**Given** domain-valid Hypothesis strategies for counts, templates, bounds,
periods, and cascades, **when** properties execute, **then** shrinking remains
enabled, the fixed CI seed reproduces failures, and minimal failures become
permanent example regressions. `[NFR-002; PBT-01, PBT-03, PBT-07, PBT-08, PBT-10]`

#### Scenario: Compatibility gates

**Given** all new focused tests pass, **when** the feature is completed, **then**
architecture, Ruff, existing non-solver, packaging, API, serialization, and
documentation gates remain green with no new mandatory dependency or blocking
PBT finding. `[NFR-004, NFR-005; PBT-01 through PBT-10]`

## Requirements Coverage Matrix

| Requirement | Stories |
|---|---|
| FR-001 Public workflow | UPO-02, UPO-09, UPO-10 |
| FR-002 Objective selection | UPO-02, UPO-04, UPO-05 |
| FR-003 Level counts | UPO-01, UPO-05 |
| FR-004 Templates | UPO-01, UPO-03, UPO-04, UPO-05 |
| FR-005 Bounds and initial candidates | UPO-01, UPO-08, UPO-11 |
| FR-006 Ordering and separation | UPO-01, UPO-03, UPO-08 |
| FR-007 Direct and Total Site scope | UPO-02 |
| FR-008 Multiperiod feasibility and coverage | UPO-03, UPO-06 |
| FR-009 Thermodynamic evaluation | UPO-04 |
| FR-010 Monetary evaluation | UPO-05 |
| FR-011 Deterministic optimisation | UPO-07, UPO-08, UPO-11 |
| FR-012 Constraint handling | UPO-03, UPO-08, UPO-10 |
| FR-013 Detached result | UPO-02, UPO-09, UPO-10 |
| FR-014 Typed result | UPO-02, UPO-04, UPO-05, UPO-06, UPO-07, UPO-09 |
| FR-015 Reporting | UPO-07 |
| FR-016 Batch isolation | UPO-10 |
| FR-017 Executable notebook example | UPO-02, UPO-05, UPO-12 |
| NFR-001 through NFR-007 | UPO-01 through UPO-12, with primary ownership in UPO-08, UPO-09, UPO-11, and UPO-12 |

## PBT Traceability Matrix

| PBT rule | Stories carrying the future obligation |
|---|---|
| PBT-01 Property identification | UPO-12 |
| PBT-02 Round-trip | UPO-01, UPO-09 |
| PBT-03 Invariants | UPO-01, UPO-03, UPO-04, UPO-06, UPO-08, UPO-09, UPO-12 |
| PBT-04 Idempotency | UPO-12; applicability to normalization is decided in Functional Design |
| PBT-05 Oracle/model testing | UPO-04, UPO-11 |
| PBT-06 Stateful testing | UPO-10; applicability to detached batch coordination is decided in Functional Design |
| PBT-07 Generator quality | UPO-09, UPO-11, UPO-12 |
| PBT-08 Shrinking and reproducibility | UPO-11, UPO-12 |
| PBT-09 Framework selection | UPO-12; Hypothesis is already selected |
| PBT-10 Complementary strategy | UPO-12 |

## Persona Mapping

| Story | Personas |
|---|---|
| UPO-01 | P-01 primary; P-02 supporting |
| UPO-02 | P-01 primary; P-02 supporting |
| UPO-03 | P-01 primary; P-03 reviewer |
| UPO-04 | P-01 primary; P-03 reviewer |
| UPO-05 | P-01 primary; P-02 integrator; P-03 reviewer |
| UPO-06 | P-01 primary; P-03 reviewer |
| UPO-07 | P-01 primary; P-02 integrator; P-03 reviewer |
| UPO-08 | P-01 primary; P-02 integrator; P-03 reviewer |
| UPO-09 | P-02 primary; P-04 maintainer |
| UPO-10 | P-02 primary; P-01 supporting; P-04 maintainer |
| UPO-11 | P-03 primary; P-04 maintainer; P-01 beneficiary |
| UPO-12 | P-04 primary; P-03 reviewer; all personas beneficiaries |

## INVEST Verification

| Story | I | N | V | E | S | T | Verification rationale |
|---|---|---|---|---|---|---|---|
| UPO-01 | Pass | Pass | Pass | Pass | Pass | Pass | One bounded inventory contract with explicit validation outcomes. |
| UPO-02 | Pass | Pass | Pass | Pass | Pass | Pass | One callable journey with two required scope scenarios. |
| UPO-03 | Pass | Pass | Pass | Pass | Pass | Pass | One independently checkable demand-conservation outcome. |
| UPO-04 | Pass | Pass | Pass | Pass | Pass | Pass | One thermodynamic objective with analytical acceptance evidence. |
| UPO-05 | Pass | Pass | Pass | Pass | Pass | Pass | One monetary objective and explicit cogeneration decomposition. |
| UPO-06 | Pass | Pass | Pass | Pass | Pass | Pass | One shared-placement multiperiod behavior with explicit identities. |
| UPO-07 | Pass | Pass | Pass | Pass | Pass | Pass | One result-inspection outcome with bounded reporting criteria. |
| UPO-08 | Pass | Pass | Pass | Pass | Pass | Pass | One failure-recovery value with typed observable outcomes. |
| UPO-09 | Pass | Pass | Pass | Pass | Pass | Pass | One stable integration contract with round-trip and non-mutation checks. |
| UPO-10 | Pass | Pass | Pass | Pass | Pass | Pass | One batch outcome with order, isolation, and state criteria. |
| UPO-11 | Pass | Pass | Pass | Pass | Pass | Pass | One numerical-assurance enabler with fixed budgets and oracle tests. |
| UPO-12 | Pass | Pass | Pass | Pass | Pass | Pass | One bounded delivery-assurance enabler with explicit repository gates. |

All stories are negotiable in implementation detail while preserving their
approved user outcome and acceptance contract. No story prescribes sprint
order, staffing, or timeline.

## Extension Compliance at User Stories

- **PBT-01 through PBT-10**: N/A for enforcement at this stage. All future
  obligations are traced above, with no current blocking finding.
- **Security Baseline**: N/A; disabled.
- **Resiliency Baseline**: N/A; disabled.

## Integrated Delivery Status

- [x] UPO-01 through UPO-12 are implemented and verified through the public
  problem/workspace workflows, pure presentation, and the packaged notebook.
- [x] Direct and Total Site workflows retain physical process-stream entropy
  evidence at real temperatures and satisfy strict residual-duty coverage.
- [x] The default thermodynamic and monetary/cogeneration notebook paths execute
  from an isolated installed wheel; no utility-placement CLI was added.
- [x] The notebook uses two isothermal plus two sensible levels on each utility
  side and displays the optimized direct result on a standard GCC; the same
  detached plot surface selects a Total Site Profile for Total Site results.
