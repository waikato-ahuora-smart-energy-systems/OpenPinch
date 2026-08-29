# Utility Placement Optimisation Requirements

## Intent Analysis

- **User request**: Add a hierarchy-aware utility-placement analysis service.
  The selected zone determines whether placement uses a Process or Unit
  Operation direct GCC, a Site Total Site Profile, or a Community or Region
  indirect aggregate profile. The user may supply concise isothermal and
  sensible counts, or omit both to infer typed templates from utilities already
  specified on the `PinchProblem`. OpenPinch minimises physical entropy
  generation and returns the best utilities as a normal detached case.
  Development must follow TDD.
- **Request type**: New public analysis feature.
- **Scope estimate**: Multiple components spanning analysis, optimisation,
  domain/contracts, application accessors, case batches, tests, and one
  executable notebook.
- **Complexity estimate**: Complex numerical feature because it combines
  constrained placement, direct and Total Site targeting, multiperiod
  feasibility, balanced-composite entropy integration, deterministic
  optimisation, and typed public results.
- **Requirements depth**: Comprehensive.

## Goals

1. Accept optional isothermal and sensible level counts, independently applying
   each supplied count to both hot and cold sides; otherwise infer templates and
   counts from existing problem utilities.
2. Optimise the temperature placement of fixed utility-family templates while
   the exact ordinary Process, Total Site, or aggregate target workflow owns
   candidate duties and fallback allocation.
3. Minimise thermodynamic cost from balanced composite curves.
4. Infer direct, Total Site, or aggregate indirect targeting from the selected
   hierarchy zone instead of exposing a public base-target selector.
5. Select one placement that remains feasible across every operating period
   and minimises the period-weighted objective.
6. Return a detached normal `PinchProblem` containing the best utility set,
   without mutating the source problem, and retain typed optimisation evidence
   on the returned case.
7. Deliver the feature test-first with analytical, regression, and full
   property-based evidence.
8. Deliver one executable notebook that separately demonstrates Process/GCC and
   Site/TSP placement using two isothermal and two sensible levels per side,
   then renders each returned case with the corresponding standard plot.

## Non-Goals

- Utility duties are not independent optimizer coordinates; they are outputs
  of the exact ordinary target replay for each candidate and period.
- Utility technology and fluid identity are not chosen by the optimiser.
- Hybrid weighted objectives and Pareto-frontier generation are excluded.
- Monetary, boiler, turbine, fuel-cost, electricity-credit, and capital-cost
  optimisation are deferred in full.
- Optimised utilities are not written into the source problem or automatically
  registered in a workspace.
- A dedicated ``problem.plot.utility_placement(...)`` API is excluded; users
  shall use ordinary named-case, targeting, and plotting workflows.
- A new package-root export, server, database, external API, or infrastructure
  component is not required.
- CLI integration is excluded.
- Additional utility-placement notebooks beyond the one named in FR-017 are
  excluded from this release.
- A full evaluation trace is not part of the normal result contract.

## Terminology and Mathematical Model

### Utility-level counts

Let `N_iso` be the `isothermal` count and `N_sens` be the `sensible` count
supplied by the caller. Both counts shall be applied independently to the hot
and cold utility sides. The service shall therefore create or validate:

- `N_iso` hot and `N_iso` cold isothermal templates; and
- `N_sens` hot and `N_sens` cold sensible templates.

`N_iso` shall be an integer of at least two. `N_sens` shall be a non-negative
integer and may be zero. The total number of utility levels on each side is
`N_total = N_iso + N_sens`.

When both counts are omitted, existing input utilities become placement
templates. Ordinary utilities with no target temperature, or with an absolute
supply-to-target span within the configured isothermal tolerance, are
isothermal; larger spans and segmented or multi-point profiles are sensible.
`Hot` and `Cold` utilities retain their side. A `Both` utility produces paired
hot and cold templates with opposite temperature direction. If one side has
fewer inferred templates of a kind, deterministic generic templates pad that
side to the larger inferred count. The inferred result must still contain at
least two isothermal templates per side.

Supplying either count deliberately selects generated-template mode and
discards existing utilities as placement templates. `isothermal` is then
required and must be at least two; omitted `sensible` means zero. Each
conceptual generated level produces one hot and one cold utility entry. Hot
utility target temperature is below supply temperature, while cold utility
target temperature is above supply temperature.

The two generated entries of the same kind and ordinal are one coupled
temperature level. Their endpoints shall be exact reversals:
`cold_supply = hot_target` and `cold_target = hot_supply`. This applies to
isothermal and sensible generated levels. The pair shares temperature decision
coordinates, while hot and cold duties remain independent and may separately
be zero. Generated pairs are ordered together from hottest to coldest; the
independent ascending-cold ordering rule does not apply within generated pairs.
Unrelated inferred Hot and Cold utilities remain independent.

### Utility templates

Each level template shall have a stable identity and define:

- hot or cold role;
- fluid and phase metadata when available;
- heat-transfer and delta-temperature-contribution metadata needed by the
  existing targeting service;
- supply-temperature bounds;
- utility `delta_temperature` bounds or a fixed value;
- an isothermal flag or equivalent fixed-span declaration.

The template fixes identity and non-temperature attributes. It does not fix a
sensible utility's solved temperatures unless explicit fixed bounds are used.

### Utility duty upper bounds

The public workflow shall accept an optional `maximum_duties` mapping keyed by
the globally unique generated or inferred utility name. A scalar limit applies
to every selected period; existing scalar-with-unit and period-resolved value
forms are accepted and normalized to the configured heat-flow unit. Limits
shall be finite and non-negative. An omitted name is unbounded and a zero
limit disables duty on that named level. Unknown names, duplicate normalized
names, missing period values, incompatible units, and negative or non-finite
limits shall fail before optimizer execution.

For each named level `i` and selected period `p`, allocation shall enforce
`0 <= Q_utility[i,p] <= Q_max[i,p]`.

Hot and cold members of a generated temperature pair retain independent duty
limits. The limit is capacity metadata, not the allocated `heat_flow`, and the
returned normal case shall retain it so subsequent ordinary targeting and
standard plots respect the same capacity constraint.

### Temperature decision variables

For each sensible level `i`, the optimiser shall choose:

- supply temperature `T_supply[i]`; and
- non-negative utility span `delta_temperature[i]`.

The target temperature shall be derived as:

- hot utility: `T_target = T_supply - delta_temperature`;
- cold utility: `T_target = T_supply + delta_temperature`.

For an isothermal level, `delta_temperature` shall be fixed to a configurable
near-zero value. The default shall be `0.01 degC`, matching OpenPinch's existing
minimum sensible-span convention. The public contract shall describe this as a
numerical representation of an isothermal level, not a physical temperature
glide.

### Candidate duty allocation

For every candidate placement and period, the service shall construct a
detached ordinary case containing the candidate temperatures and the caller's
utility capacity limits, then invoke the same public target workflow the
returned case will run. A Process Zone or Unit Operation shall use direct
targeting. A Site shall run its child direct targets and ordinary Total Site
aggregation. A Community or Region shall run its ordinary indirect aggregate
target. The resulting named and fallback duties are candidate outputs, not an
independent stick-breaking decision vector.

Temperature decisions remain shared across selected periods. Targeted duties
remain period-specific because process and site profiles vary by period. Hot
and cold members of a generated reversed-temperature pair retain independent
duties. Any candidate level may receive zero duty; requesting a level makes it
available but does not impose a lower duty bound.

For every candidate `c` and period `p`, let `Q_heat_required[c,p]` and
`Q_cool_required[c,p]` be the non-negative targets produced by that exact
ordinary replay. This candidate dependence is required for Total Site cases,
where matching utility generation and use can change net hot and cold utility
targets. The coverage constraints shall require, within one named numerical
tolerance:

- `sum(Q_hot_utility[i,c,p]) = Q_heat_required[c,p]`; and
- `sum(Q_cold_utility[i,c,p]) = Q_cool_required[c,p]`.

No feasible candidate may leave unmet heating or cooling demand, and no excess
allocation beyond the same tolerance may be silently accepted. Individual
levels may have zero duty when the complete side-level balance still holds.

Each targeted level duty shall be limited by both its caller-specified maximum
duty, when present, and the duty available through the exact target workflow at
its selected temperature interval. Residual `HU` or `CU` duty shall remain an
explicit fallback and shall receive the dimensionless squared fallback penalty.

When capped named levels cannot cover a side's complete residual demand, the
generated `HU` or `CU` default utility shall supply only the remaining
shortfall. A default is not a requested or inferred placement level and shall
never displace available named capacity. It shall be retained in the returned
case and standard plots when its duty is positive in any selected period.

### Thermodynamic cost

Thermodynamic cost shall be physical entropy generation calculated from the
candidate's balanced hot and cold composite curves at real temperatures. The
balanced curves shall include the process composites plus only the candidate's
allocated utility streams and shall be aligned on their union heat-load grid.
For each matched heat-transfer interval, sensible entropy change shall use
`CP * ln(T_out / T_in)` in kelvin and an isothermal interval shall use the
signed `Q / T` limit. Total entropy generation shall be the cold-composite
entropy gain plus the hot-composite entropy change, summed over all intervals.

Moving a feasible utility temperature closer to its matched process
temperature shall reduce entropy generation. The implementation shall reject
non-positive absolute temperatures, materially negative entropy generation,
and unbalanced composite duties. Exergy destruction shall equal ambient
absolute temperature multiplied by physical entropy generation.

### Deferred monetary scope

Utility-placement contracts shall not expose an objective selector, prices,
cogeneration eligibility, electricity-price or turbine inputs, monetary result
fields, or placement-specific monetary evaluators. A future monetary feature
requires a separate approved boiler/turbine requirements and design cycle.

### Multiperiod objective

The same utility template ordering and solved temperature variables shall be
used for every period. A candidate is infeasible if it is infeasible in any
period. For objective `C_p` and existing non-negative period weight `w_p`, the
aggregate objective shall be `sum(w_p * C_p)`. At least one period weight must
be positive. Per-period values and the aggregate shall both be returned.

Default fallback use shall be discouraged by a squared, dimensionless penalty.
For period `p`:

`g_penalty[p] = (Q_HU[p] / Q_heat_required[p])^2 + (Q_CU[p] / Q_cool_required[p])^2`.

Each side term is defined as zero when both its required and fallback duties
are zero; positive fallback duty against zero required duty is invalid.

The aggregate fallback penalty shall be `sum(w_p * g_penalty[p])` and shall be
combined monotonically with the existing dimensionless optimizer scalar while
preserving the feasible/infeasible ranking partition. The reported
physical objective remains the period-weighted entropy generation in `kW/K`;
fallback penalties shall be reported separately and shall not be presented as
physical entropy.

## Functional Requirements

### FR-001: Public workflow

The primary workflow shall be
`problem.target.utility_placement(isothermal=..., sensible=..., zone=...,
maximum_duties=...)`.
All three arguments are optional when existing utilities permit inference. It shall
return a detached `PinchProblem` whose utility input contains only the best
optimized utility set. It shall be mirrored by supported workspace case-batch
and all-period surfaces in the same style as other target operations.
Specialist types and services shall be importable from their concrete owner
modules, without expanding the two-symbol package-root facade.

### FR-002: Thermodynamic-only objective

The workflow shall always minimize thermodynamic entropy generation. It shall
not expose an objective selector or accept monetary placement inputs.

### FR-003: Level-count validation

The workflow shall accept optional `isothermal` and `sensible` inputs. If
either is supplied, generated-template mode applies both counts to both sides,
requires `isothermal` of at least two, and treats omitted `sensible` as zero.
If both are omitted, the counts and templates shall be inferred from existing
utilities. The previous `_level_count` public keywords, public template
collections, and public base-target selector shall not remain in the supported
signature. Booleans, non-integral values, negative values, missing generated
isothermal count, or an inferred inventory with fewer than two isothermal
levels per side shall raise a typed, actionable error before targeting or
optimizer execution.

### FR-004: Template validation

Generated mode shall create exactly `N_iso` hot and `N_iso` cold isothermal
templates plus `N_sens` hot and `N_sens` cold sensible templates. Inferred mode
shall classify ordinary, segmented, and profiled utilities by thermal span,
expand `Both` utilities to paired opposite-side templates, preserve typed
utility identity and fluid metadata, and deterministically pad a smaller side
to the larger count for each kind.
Names shall be unique within a placement request. Template kind shall be
explicit and shall agree with fixed or variable `delta_temperature` behavior.
Bounds, spans, approach metadata, and optional fluid data shall be
finite and dimensionally valid. Hot and cold roles shall agree with their
temperature direction.
Matching generated templates shall share one supply/span decision and decode
to exact endpoint reversals without duplicate optimizer coordinates.
Maximum-duty names shall resolve against this final generated or inferred
inventory and shall be independently validated and normalized per selected
period.

### FR-005: Bounds and starting candidates

Default supply-temperature bounds shall cover the intervals where the selected
shifted residual hot and cold profiles change; generated levels shall not all
be forced outside the global process-temperature extremes. Sensible span
bounds shall cover the complete target temperature range. The effective model
shall retain minimum approach constraints and the intersection of feasible
bounds across all selected periods. Callers may provide narrower overrides but
may not expand beyond physical feasibility. The service shall produce
deterministic valid initial candidates when the feasible region is nonempty.
Structured starts shall distribute both supply temperatures and sensible spans
across their effective intervals so edge clustering or a midpoint-only seed
cannot mask a closer profile fit.

### FR-006: Ordering and separation

The combined isothermal and sensible levels shall be strictly descending on the
hot side and strictly ascending on the cold side by supply temperature.
Adjacent supply temperatures shall satisfy a configurable positive minimum
separation regardless of template kind. Defaults and overrides shall use
OpenPinch's temperature-difference unit handling. Coincident levels shall not
be merged after optimisation.

### FR-007: Direct and indirect scope

The public API shall not expose `base_target`. If `zone` is omitted, placement
uses the `PinchProblem.master_zone`. If supplied, a zone name, unique name, full
zone address, or `Zone` object shall resolve to a node in that problem's
hierarchy. Process Zone and Unit Operation nodes use direct GCC targeting; Site
nodes use Total Site targeting; Community and Region nodes use indirect
aggregate targeting. Utility Zone is unsupported. Missing, ambiguous,
foreign-object, or incompatible zone selections shall fail with actionable
guidance before optimization.

### FR-008: Multiperiod feasibility

One placement shall be replayed independently against every ordered period.
Period weights and identities shall use the existing canonical problem context.
No weighted-average surrogate process profile shall replace per-period
feasibility. In every period, allocated hot utilities shall cover the complete
residual heating demand and allocated cold utilities shall cover the complete
residual cooling demand within the named coverage tolerance. A coverage failure
on either side in any period makes the candidate infeasible before objective
ranking unless the residual-only default `HU` or `CU` fallback covers that
shortfall. Every named duty shall satisfy its period-specific maximum within
the same numerical tolerance.

### FR-009: Thermodynamic evaluation

The thermodynamic evaluator shall calculate total entropy generation,
utility-side entropy changes, process-side entropy changes, and exergy
destruction for every period and aggregate. It shall work for sensible and
isothermal balanced-composite intervals without division-by-zero or
logarithmic cancellation.

### FR-010: Monetary capability exclusion

Placement-specific monetary and cogeneration evaluators, settings, fields, and
result breakdowns shall be absent. Existing general OpenPinch cogeneration
analysis shall remain unchanged.

### FR-011: Deterministic optimisation

The service shall use the existing solver-neutral bounded minimisation service.
The default method shall be `cmaes`. The default seed, tolerances, maximum
evaluations, and candidate limit shall be deterministic, and every setting
shall be overridable through typed or validated options. Exact per-call method
choices shall remain `dual_annealing`, `cmaes`, `bo`, and `rbf_surrogate`. The
same request, environment, and options shall produce equivalent ordered results
within documented numerical tolerances.

### FR-012: Constraint handling

Individual infeasible candidates may be rejected or assigned a deterministic
penalty that cannot outrank a feasible candidate. If no feasible candidate is
found, the service shall raise a typed error containing the scope,
isothermal and sensible level counts, period context, unmet heating or cooling
duty, and useful constraint diagnostics. It shall not return a least-infeasible
placement as success.

Automatically generated default utilities (`HU` and `CU`) are balancing
fallbacks, not placement options. They shall be considered only after capped
named utilities have exhausted their available duty. Positive fallback duty
remains feasible, contributes to the balanced-composite entropy calculation,
and receives the deterministic squared `g_penalty()` ranking term specified
above. Fallback duty and penalty evidence shall be explicit rather than hidden
as a coverage residual.

### FR-013: Detached optimized case

Execution shall not mutate the problem's process streams, utilities,
configuration, targets, selected case, or cached study inputs. Any internal
candidate objects shall be detached. The public return shall be a new
`PinchProblem` built from the source input with its utility collection replaced
by the best optimized hot and cold utility levels. It shall behave like a
normal unsolved case: callers explicitly run ordinary targeting and plotting.
Per-utility maximum-duty metadata and any required default fallback definitions
shall survive this replacement so ordinary retargeting reproduces capped
allocation behavior. The source problem remains unchanged.

### FR-014: Placement evidence on the returned case

The returned optimized case shall retain a detached
`utility_placement_result` containing:

- request metadata, inferred scope, hierarchy-zone identity, isothermal and
  sensible level counts, total levels per side, method, seed, termination
  metadata, and units;
- one best feasible solution;
- deterministically ordered alternative feasible candidates;
- hot and cold utility templates with solved supply/target temperatures and
  allocated duties and optional maximum duties;
- per-period and aggregate objective decomposition;
- entropy generation and exergy destruction;
- per-period default utility duties and squared fallback penalties;
- feasibility status and non-fatal diagnostics.

The evidence and nested candidates shall round-trip through the supported JSON
contract without exposing callables, mutable runtime streams, or
backend-private objects. Normal users shall not need to traverse this evidence
to obtain utilities, create a case, target it, summarize it, or plot it.

### FR-015: Reporting integration

Placement evidence shall remain available through the returned case's
`utility_placement_result`. Ordinary case metrics, summaries, reports, targets,
and plots shall use the same methods as every other `PinchProblem`; the former
placement-specific metrics, frame, and report methods shall be removed.
`PinchWorkspace` shall add
`workspace.add(case, *, name, activate=False)` for explicit registration of a
normal case while preserving its placement evidence. No placement-specific
plot method shall be added. The notebook shall show optimization, workspace
registration, normal targeting, and standard GCC and Total Site Profile plots
without manually constructing utility dictionaries.

### FR-016: Batch isolation

Workspace case batches shall preserve case order and isolate per-case failures
using existing batch-result conventions. A failure in one case shall not mutate
or suppress successful results from other cases.

### FR-017: Executable notebook example

Exactly one new utility-placement notebook shall be delivered at
`OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`. It shall be
owned by the repository notebook generator, registered in the canonical
tutorial manifest, included in source and wheel distributions, and executable
from top to bottom under the appropriate notebook dependency profile. Using
only public Python APIs, one coherent example shall demonstrate:

- one Process placement and one Site placement, each with two isothermal and
  two sensible hot utility levels plus their two isothermal and two sensible
  cold counterparts;
- the default thermodynamic or entropy-based placement workflow;
- inspection of entropy-generation and exergy-destruction result breakdowns;
- receipt of the best optimized utilities as a normal detached case followed
  by `workspace.add(...)` registration under a new name;
- normal direct targeting and standard GCC plotting of the Process result, plus
  normal Total Site targeting and standard Total Site Profile plotting of the
  Site result, with utilities visible in both; and
- deterministic bounded options and inspectable comparison tables showing
  optimizer-evidence and ordinary-retarget duties side by side, without test
  assertions in the notebook; and
- at least one concise `maximum_duties` example that verifies a named cap and
  shows residual default utility duty and `g_penalty()` evidence when required.

No CLI command or CLI invocation shall be added for utility placement.

## Non-Functional Requirements

### NFR-001: Numerical correctness

All thermodynamic calculations shall use absolute temperature internally,
units-aware boundaries, stable logarithmic or limiting formulas, and explicit
finite-value checks. Tolerances shall be named and tested rather than embedded
as unexplained constants.

### NFR-002: Reproducibility

Default and CI execution shall use fixed seeds. Candidate ordering shall use
objective value followed by a deterministic coordinate tie-breaker. Failure
reports shall retain enough method and seed metadata to reproduce the run.

### NFR-003: Bounded execution

The public options shall bound iterations and objective evaluations. Focused
tests shall use small deterministic budgets. No universal wall-clock guarantee
is imposed because backend and study complexity vary, but the service shall
honour evaluation budgets and return or fail without unbounded retry loops.

### NFR-004: Compatibility

The change shall preserve the package-root API, existing target behavior,
serialization schemas, units, and optional-dependency boundaries. It shall add
no new mandatory runtime dependency. Because utility placement is new and not
yet released, the verbose `isothermal_level_count` and
`sensible_level_count` keywords, public template collections, `base_target`
selector, and detailed-result return are replaced rather than retained as
competing public workflows.

### NFR-005: Maintainability

Pure kernels shall own template normalization, vector encoding/decoding,
constraint checks, entropy calculation, and objective composition. Application
accessors shall coordinate rather than contain numerical equations.

### NFR-006: Observability

Typed diagnostics shall distinguish input validation, empty feasible bounds,
candidate infeasibility, targeting failure, non-finite objective, and
optimiser exhaustion without broad exception suppression.

### NFR-007: Security and resiliency scope

No new network, file, credential, or remote-execution boundary is introduced.
The Security and Resiliency extensions are disabled for this feature. Existing
repository safety and validation conventions remain applicable.

## TDD and Acceptance Requirements

Implementation shall use red-green-refactor sequencing. Each production slice
must begin with a failing focused test, add the smallest implementation that
makes it pass, and refactor only with the focused tests green.

### Example-based acceptance tests

1. Reject `N_iso < 2`, `N_sens < 0`, non-integral counts, and mismatched
   isothermal or sensible template counts before service execution.
2. Encode and decode mixed sensible and near-isothermal templates with the
   correct number and order of decision variables.
3. Derive hot and cold target temperatures from supply temperature and span.
4. Reproduce hand-calculable constant-CP entropy-generation examples, including
   the near-isothermal limit and both hot and cold utilities.
5. Demonstrate that lower entropy placement wins a two-level analytical case.
6. Prove the public request, template, result, accessor, and notebook surfaces
   contain no monetary or placement-specific cogeneration capability.
7. Prove one shared placement is evaluated against every period and uses the
   raw configured weighted sum.
8. Prove exact heating and cooling coverage within tolerance for every period,
   including a case where an individual level receives zero duty.
9. Prove a candidate with unmet heating or cooling demand in one period cannot
   become the best aggregate solution.
10. Exercise direct and Total Site scopes on existing sample cases.
11. Prove optimization returns a detached normal case with the best utility
    set and does not mutate source problems, utilities, targets, or cases.
12. Round-trip retained placement-evidence contracts through JSON.
13. Re-run an identical fixed-seed request and compare ordered candidates.
14. Verify typed diagnostics for invalid bounds, non-positive Kelvin values,
    unmet demand, and no feasible candidates.
15. Verify workspace batch ordering and failure isolation.
16. Execute `19_utility_placement_optimisation.ipynb` from a clean generated
    artifact and verify the thermodynamic workflow, public-API-only imports,
    manifest registration, package inclusion, and absence of test assertions.
17. Verify that mixed two-isothermal plus two-sensible placement returns a
    normal case, `workspace.add(...)` registers it without changing the source
    or active case by default, and normal targeting renders its standard GCC
    and Total Site Profile with utilities without a placement-specific plot
    API.
18. Prove with analytical balanced-composite pairs that zero driving force has
    zero entropy generation, that increasing the matched temperature approach
    increases cost, and that the optimizer ranks the closer feasible utility
    placement first. Verify the executable notebook no longer selects extreme
    utility temperatures visually separated from the residual process GCC.
19. Reproduce hand-calculable balanced-composite sensible and isothermal
    entropy-generation cases containing the expected logarithmic and `Q/T`
    terms. Prove that residual-only `HU` or `CU` duty contributes physical
    entropy and a separate squared `g_penalty()` term.
20. Prove the public signature uses only `isothermal` and `sensible`, the call
    returns only the optimized `PinchProblem`, placement evidence remains on
    that case, and the notebook contains no nested best-period traversal or
    manual utility-dictionary construction.
21. Prove omitted counts infer ordinary isothermal and sensible utilities,
    expand `Both` with reversed temperature direction, pad asymmetric inferred
    sides deterministically, and preserve source input unchanged.
22. Prove explicit counts override existing utilities and produce paired hot
    and cold entries whose temperatures are exact endpoint reversals and whose
    duties remain independent, while a partially specified generated request
    without `isothermal` fails before analysis.
23. Prove master-zone defaulting and explicit name, path, and object selection;
    verify Process/Unit Operation direct, Site Total Site, and Community/Region
    indirect routing plus typed ambiguous, foreign, missing, and Utility Zone
    errors.
24. Prove `maximum_duties` accepts scalar, explicit-unit, and period-resolved
    values, broadcasts scalars, and rejects unknown names, invalid units,
    negative/non-finite limits, and incomplete selected-period data before the
    optimizer runs.
25. Prove each named allocation stays at or below its own cap in every selected
    period, omitted names remain unbounded, zero disables the named level, and
    generated hot/cold pair caps remain independent.
26. Prove the targeting allocator exhausts available named utility capacity
    before allocating residual `HU` or `CU` duty, and that squared fallback
    penalties aggregate with raw period weights without altering reported
    physical entropy units.
27. Prove the returned detached case retains maximum-duty metadata and any
    required fallback so ordinary Process GCC and Site TSP targeting and plots
    reproduce capped allocation without source mutation.

### Property-based acceptance tests

Full Property-Based Testing enforcement is enabled. Hypothesis shall generate
domain-valid templates, ordered bounds, positive absolute temperatures,
separate valid isothermal and sensible counts, periods, weights, and feasible
small cascades. Required properties include:

- vector encode/decode round-trip;
- result JSON round-trip within declared floating-point tolerances;
- hot-descending and cold-ascending ordering;
- target-temperature direction and bound preservation;
- fixed near-isothermal span preservation;
- non-negative feasible entropy generation within tolerance;
- exergy destruction equals ambient absolute temperature times entropy
  generation;
- feasible candidates always outrank penalised infeasible candidates;
- objective and result invariance under detached input copying;
- deterministic result ordering and fixed-seed reproducibility;
- optimiser results checked by a brute-force or structured-grid oracle on
  bounded small analytical cases;
- weighted objective equals the explicit sum of per-period contributions;
- allocated hot and cold utility duties conserve and completely cover their
  respective residual demands within tolerance in every feasible period;
- serialization never introduces backend-private or mutable runtime state.
- inferred utility classification, `Both` expansion, side padding, and zone
  routing are deterministic under detached input copies and declaration order.
- every named allocation is bounded by its scalar or period-resolved maximum;
- squared fallback penalty is non-negative, zero exactly at zero fallback,
  monotone with fallback magnitude, scale invariant when fallback and required
  duties are scaled together, and equals the explicit weighted-period sum.

Shrinking shall remain enabled. The repository's fixed Hypothesis seed shall be
used in CI, and any newly discovered minimal counterexample shall become a
permanent example-based regression test.

## Requirement Traceability

| Decision source | Requirements |
|---|---|
| Initial request | Goals 1-7; FR-002; TDD requirements |
| Initial Q1-Q6 | FR-003 through FR-006; utility count and variables |
| Initial Q7-Q11 | Historical objective discussion; superseded by the thermodynamic-only amendment |
| Initial Q12-Q13 | FR-013; FR-014 |
| Initial Q14-Q18 | FR-007; FR-008; FR-011; FR-012; TDD evidence |
| Initial Q19-Q21 | NFR-007; full PBT acceptance requirements |
| Clarification Q1 and Q4 | Temperature decision variables and near-isothermal default |
| Clarification Q2 | Thermodynamic cost integration |
| Clarification Q3 | Fixed template identity and eligibility |
| Clarification Q5 | Per-period feasibility and weighted-sum objective |
| Requirements change request | Separate `N_iso` and `N_sens` inputs; minimum two isothermal levels; complete heating and cooling coverage |
| Notebook scope decision | Goal 8; FR-017; Acceptance 16; CLI explicitly excluded |
| Balanced-composite entropy correction | FR-009; FR-012; Acceptance 18-19; logarithmic/`Q/T` entropy and generated `HU`/`CU` exclusion |
| Thermodynamic-only scope correction | Goal 3; Goal 8; FR-002; FR-004; FR-010; FR-014; FR-017; Acceptance 6 and 16 |
| Hierarchy API answers and clarifications | Goals 1, 4, and 8; FR-001; FR-003; FR-004; FR-007; FR-014; FR-017; Acceptance 21-23 |
| Maximum-duty answers and fallback clarification | Utility duty upper bounds; FR-001; FR-004; FR-008; FR-009; FR-012 through FR-017; Acceptance 24-27 |
| Profile-aware dispatch correctness approval | Goal 2; Candidate duty allocation; FR-008; FR-009; NFR-001 through NFR-003; Acceptance 18, 25-27; profile-aware dispatch amendment |
| Result correctness answers and clarification | Exact ordinary-target replay; candidate-only levels; no minimum duty; FR-008 through FR-017; Acceptance 28-31 |

## Success Criteria

The feature is complete when all functional and non-functional requirements are
implemented, all new example and property tests pass with the fixed seed, the
existing non-solver suite and architecture gates remain green, direct and Total
Site sample regressions pass, result serialization and batch behavior are
verified, documentation demonstrates the thermodynamic workflow, and no enabled PBT
rule has a blocking finding. The single required utility-placement notebook
shall execute cleanly, demonstrate balanced-composite entropy placement, and
be present in the tutorial manifest and built distributions. Every successful
candidate shall also pass the per-period heating and cooling coverage
equalities within tolerance.

## Extension Compliance at Requirements Analysis

| Rule | Status | Rationale |
|---|---|---|
| PBT-01 | N/A at this stage | Formal property identification is enforced during Functional Design; required categories are anticipated above. |
| PBT-02 | N/A at this stage | Round-trip implementation is not present yet; it is an explicit acceptance requirement. |
| PBT-03 | N/A at this stage | Ordering, bounds, feasibility, and objective invariants are specified for later tests. |
| PBT-04 | N/A at this stage | No idempotent production operation is designed yet. |
| PBT-05 | N/A at this stage | A small brute-force or grid oracle is required during construction. |
| PBT-06 | N/A at this stage | The service is required to be detached and stateless; application batch state will be assessed during design. |
| PBT-07 | N/A at this stage | Domain generator requirements are specified for construction. |
| PBT-08 | Compliant | Fixed-seed reproducibility, shrinking, and regression capture are required. |
| PBT-09 | Compliant | Existing Hypothesis and pytest are selected and already declared dependencies. |
| PBT-10 | Compliant | Both analytical example tests and complementary property tests are mandatory. |

Security Baseline and Resiliency Baseline are disabled by explicit user choice,
so their rules are skipped for this stage.

## Construction Verification Status

- [x] FR-001 through FR-017 are implemented and covered by specialist,
  application, presentation, notebook, documentation, architecture, and
  distribution tests.
- [x] NFR-001 through NFR-007 are satisfied by analytical and oracle evidence,
  fixed-seed repeatability, bounded performance tests, compatibility and import
  gates, typed diagnostics, and confirmation that no new external boundary was
  introduced.
- [x] Acceptance requirements 1 through 17 pass, including direct and Total
  Site samples, all-period feasibility, source/case non-mutation, ordered batch
  isolation, JSON round-trip, mixed-level replacement in a new named case,
  normal GCC/TSP rendering, absence of a placement-specific plot API, and the
  installed-wheel notebook workflow.
- [x] PBT-01 through PBT-10 are compliant at construction completion; the final
  findings are recorded in each unit code summary and the integrated Build and
  Test summary.
- [x] Exactly one utility-placement notebook is packaged and executable, and no
  utility-placement CLI surface was added.
- [x] The packaged direct and Total Site samples use period-resolved physical
  process streams at real temperatures for process entropy; the direct result
  has nonzero process entropy and both scopes meet exact residual coverage.
- [x] The maximum-duty amendment completed TDD implementation and integrated
  Build and Test for Acceptance 24 through 27.
- [x] The profile-aware dispatch correction adds conserving per-period duty
  decisions, scalar-consistent ranking, entropy-unit scaling, bounded ordered
  optimizer coordinates, and current Process/GCC and Site/TSP notebook evidence.
- [x] The exact-target replay correction removes independent duty coordinates,
  derives duties from ordinary hierarchy-aware targeting, and shows their
  equality through assertion-free Process/GCC and Site/TSP notebook tables.

## Profile-aware dispatch correctness amendment

The optimizer shall rank candidates using the same complete scalar objective
used by its backend: normalized balanced-composite entropy plus the separately
reported fallback penalty. Canonical replay shall not reorder candidates using
unpenalized entropy alone.

The entropy normalization scale shall have entropy units. It shall be derived
from deterministic baseline placement entropy or another positive
context-derived entropy reference, never directly from heat duty. Scaling must
remain constant for every candidate in one optimization run.

Optimizer termination evidence shall include canonical parent-process replay
evaluations and shall not report zero evaluations after successful candidate
replay.

The exact-target replay amendment supersedes the independent duty-coordinate
part of this section. Candidate levels may remain inactive. Standard GCC and
Total Site plots remain the visual acceptance evidence, and their retargeted
duties shall match optimizer evidence by utility name and fallback status.

## Exact ordinary-target replay amendment

The optimizer shall evaluate the exact hierarchy-aware public target workflow,
not a simplified aggregate load-profile allocator. Candidate thermodynamic
inputs, net utility targets, named duties, fallback duties, and Total Site
utility-to-utility matching shall all come from that detached replay.

The returned case shall retain only inputs that ordinary targeting can replay.
Running the same scope and period on that case shall reproduce optimizer
evidence for every named and fallback utility within the declared tolerance.
The standard GCC or Total Site Profile shall therefore visualize the allocation
that was actually ranked. Independent period duty coordinates that cannot be
reproduced through the normal case workflow shall be removed.

Additional acceptance requirements:

28. Prove Process optimizer evidence and ordinary direct retargeting agree by
    utility name, duty, fallback status, and target totals.
29. Prove Total Site optimizer evidence and ordinary Total Site retargeting
    agree after child direct targeting, utility aggregation, and same-level
    utility generation/use matching.
30. Prove candidate-specific Total Site targets and balanced-composite entropy
    are calculated from the exact replay rather than baseline residual duties or
    a simplified aggregate-profile allocation.
31. Prove requested levels may have zero duty, no implicit minimum-duty bound is
    introduced, and source cases remain unchanged during every replay.

Final correction evidence:
`aidlc-docs/construction/utility-placement-exact-target-replay/build-and-test/build-and-test-summary.md`.

## Utility-profile non-crossing amendment

Every ordinary Process utility target and every utility-placement candidate
replayed through that target shall be thermodynamically feasible over the whole
temperature range, not only at its endpoints. At every Process GCC breakpoint,
the cumulative sensible utility profile shall remain on the feasible side of
the residual Process GCC within the targeting tolerance. A hot utility profile
shall not exceed the hot residual envelope and a cold utility profile shall not
exceed the cold residual envelope.

The duty maximizer shall enforce the tightest endpoint or interior breakpoint
limit. It shall not discard a tighter interior limit. Multiple levels and
maximum-duty limits shall preserve the same cumulative profile invariant.

Additional acceptance requirements:

32. Preserve the notebook-derived Process counterexample as a regression: a
    hot sensible utility spanning approximately 174.01 to 62.96 degC must not
    exceed the Process GCC around 97 degC.
33. Prove with generated monotone residual profiles that assigned hot and cold
    sensible utility profiles do not cross their respective GCC envelopes at
    any breakpoint.
34. Prove normal direct targeting and utility-placement exact replay use the
    same corrected shared targeting behavior and continue to respect optional
    per-utility maximum duties.
