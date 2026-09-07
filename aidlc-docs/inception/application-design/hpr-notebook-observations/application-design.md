# HPR correction application design

Status: design approved by the user's `ok` response, including the selected
residual_utility entry point and frozen-HPR utility-placement workflow. This is
an approved design; implementation has not started. Units Generation is active.

## Decisions for review

- Add one conversion method, `problem.target.residual_utility(base_target=heat_pump)`,
  returning an unsolved detached `PinchProblem`. Continue with existing
  `target.direct_heat_integration()` for allocation and `target.utility_placement()`
  for optimization.
- Represent the frozen thermal study explicitly in canonical input through
  `residual_basis`. It carries immutable, unit-aware numerical and provenance
  data rather than live target/backend references or fabricated process streams.
  Serialization and optimized-case reconstruction preserve this basis.
- Add typed `hpr_load` and `hpr_residual` records on HPR results, distinguishing
  available, selected and achieved useful duty from total cycle/ambient duty.
- Extend HP plot methods with a target-object selector and add corresponding
  named refrigeration methods. Match cached target/period identity and never
  choose an empty placeholder or another mode's result.
- Keep allocation coordinates separate from the physical temperature/entropy
  boundary. Utility-placement evaluation consumes both verified representations;
  shifted GCC temperatures cannot be passed off as physical temperatures.
- Keep the two-class root surface. Domain owns immutable value records;
  application owns selection, provenance, case creation and lifecycle; analysis
  owns thermal computation; presentation owns rendering.

## Requirements and unit traceability

| Requirements | Design components/services | Planned units |
|---|---|---|
| R1, R3: service obligations and load meaning | C1-C2, S1; typed load reporting and separate ambient service | 1 |
| R2: economics | C2, C6; declared price roles and positive tutorial sensitivity | 1, 4 |
| R4: plots | C5, S2; target-specific transport and named refrigeration methods | 2 |
| R5: residual correctness | C1-C2, S1/S4; one verified full-precision residual and physical basis | 1, 3 |
| R6: detached utility workflow | C3-C4, S3-S5; canonical residual input and frozen candidate adapter | 3 |
| R7: tutorial and guides | C6; demonstrated complete public sequence and updated method inventory | 4 |

## Design alternatives considered

A domain target method that constructs a PinchProblem would violate inward
owner dependencies. Attaching residual arrays only to a private problem
attribute would lose them in utility placement's serialize-and-reconstruct
paths. Rebuilding arbitrary synthetic process streams from shifted profiles
would mislabel physical temperatures and entropy. The selected canonical
residual representation avoids those problems while retaining the existing
public problem, allocation and placement surfaces.

Adding a second `base_target` path directly to utility_placement is unnecessary
when the explicit residual case already supplies an unambiguous frozen input.
No alias is proposed. This also makes the intermediate residual inspectable
before optimization. The tradeoff is explicit support for a residual-only
input kind in validation, dispatch and placement; unsupported physical-stream
analyses must reject it rather than behaving as an empty process.

## Property concerns for Functional Design

| Category | Component | Required analysis |
|---|---|---|
| Invariant | C1-C2 | Duty bounds, energy closure, finite profiles, grid alignment and basis/period consistency |
| Oracle | C2/C4 | Independent physical cascade/accounting and entropy-boundary checks |
| Round-trip | C1/C3/C5 | Canonical residual input and graph transport retain engineering content and identification |
| Idempotence | C3/C5 | Repeated conversion preserves engineering content; repeat observation preserves state |
| Stateful model | C3-C5 | Resolve, convert, change source/utilities, allocate and optimize without changing the frozen basis |
| Easy verification | C4 | Every candidate's supplied basis matches the selected snapshot, and winning duties can be reallocated |

Detailed equations, absolute/relative tolerances, exact serialized field layouts,
zero-load result semantics and generators belong to Functional Design. That
stage must prove correct physical boundary construction for supported direct
and utility-system Carnot cases before they are eligible for residual placement.

## Extension compliance

| Rule | Status | Rationale |
|---|---|---|
| PBT-01 | Compliant for application design | Property categories and owners identified for detailed functional design. |
| PBT-02 | N/A at this stage | No serializers implemented; residual and graph round-trips required above. |
| PBT-03 | N/A at this stage | No algorithms implemented; duty/grid/basis invariants assigned above. |
| PBT-04 | N/A at this stage | No operations implemented; repeat conversion/observation requirements explicit. |
| PBT-05 | N/A at this stage | No numerical implementation verified; independent physical oracles required. |
| PBT-06 | Compliant for design assessment | Stateful application components identified and a frozen-basis model required. |
| PBT-07 | N/A at this stage | Generators will be specified during functional/code design. |
| PBT-08 | Compliant | Existing shrinking and seed 20260715 practices retained. |
| PBT-09 | Compliant | Existing Hypothesis/pytest framework retained. |
| PBT-10 | Compliant for design | Approved concrete stories are paired with general property categories. |
| Security Baseline | N/A | Disabled by existing user configuration. |
| Resiliency Baseline | N/A | Disabled by existing user configuration. |

No blocking extension findings at application design. This does not claim that
construction verification is complete.

## Consolidated component documents

The sections below consolidate the four required design artifacts. The separate
files remain convenient focused review views of the same design.

## HPR correction components

### C1: HPR result and residual values

Owner: a focused domain module, `OpenPinch/domain/hpr.py`, consumed by
`HeatPumpTargetBase` and transport contracts. Domain values must not import
application, presentation or analysis. Frozen models use finite scalars,
immutable tuples and explicit units rather than live Stream/Zone references.

- `HPRLoadSummary`: mode, process/utility basis, available service duty, selected
  service duty, achieved useful duty, cycle heating/cooling duties, work, direct
  ambient service and remaining external duties. Ambient source and sink
  exchanges are separate. All power magnitudes share a declared unit.
- `HPRResidualData`: verified numerical residual, physical boundary and period
  values returned by analysis before application provenance is available.
- `HPRResidualSnapshot`: the completed HPRResidualData plus source provenance
  and result digest, selected zone and
  period, effective thermal settings, original unit context, exact aligned
  residual profiles, coordinate basis, verified physical boundary data and
  frozen HPR/ambient exchange. It contains no mutable equipment/backend object.
- `HPRResidualProfile`: temperatures, net cascade and nonnegative heating- and
  cooling-utility load profiles with units and an explicit coordinate basis.
  Existing signed problem-table columns are adapted at the owner boundary.

The physical boundary data retain actual temperatures and signed thermal
contributions needed by utility-placement entropy evaluation. Shifted profiles
alone do not supply those physical values. Unit 1 owns constructing a verified
pair of allocation and physical bases; Unit 3 consumes it.

Existing HPR targets expose `hpr_load` and `hpr_residual` as read-only value
records. A zero-duty result must not fabricate an HPR snapshot. Existing shared
zero-load behavior is finalized in Functional Design. Scalar period snapshots
are distinct from weighted summaries; averaging temperature curves does not
produce an eligible residual study.

### C2: HPR numerical service

Owner: `OpenPinch/analysis/heat_pumps`, including existing load selection,
pre/postprocessing, Carnot objectives and service orchestration.

This component distinguishes service duty from total cycle and ambient duty,
validates physical feasibility, constructs consistent residuals and allocates
existing utilities. A single accounting result feeds the target summary,
graphs, residual snapshot and objective components; consumers do not invent
their own residual correction.

Shared helpers must retain explicit behavior for simulated and multiperiod HPR
callers. Carnot screening price ratios and utility stream prices keep their
existing roles; the tutorial sets explicit assumptions rather than changing
global defaults. Cycle-infeasibility penalties remain separate from economics.

### C3: Residual case application service

Owner: `OpenPinch/application/hpr_residual.py`, exposed by one thin
`PinchProblem.target.residual_utility()` method on the existing target accessor. It validates the selected local target,
deeply detaches its verified snapshot and returns another `PinchProblem`.

This is a utility study with a frozen thermal basis. It is not a reconstruction
of original process streams or a new HPR design. Source selection is resolved
once; subsequent source edits cannot alter the detached case. No solver runs
during conversion.

The derived case uses an explicit optional `TargetInput.residual_basis` field.
It has an empty process-stream list, copied utility definitions and a selected
period/zone context supplied by that field. No synthetic physical temperatures
or dummy process streams are inserted to trick the ordinary pipeline. Input
validation rejects a mixture of residual_basis and physical process streams,
network designs or process components.

Canonical JSON, case construction and the existing utility-placement result
construction preserve this basis. A private side attribute would be lost by
the current serialize-and-reconstruct candidate path and is not sufficient.

### C4: Residual utility evaluation

Owners: application targeting/utility-placement orchestration plus analysis
utility allocation and placement services.

The normal public `direct_heat_integration()` call on a residual case allocates
its utilities against the stored profile through a dedicated residual service.
It publishes a residual utility target with accurate classification and graphs;
it does not claim to solve an original process or rerun HPR. No new public
utility-allocation alias is needed.

Utility-placement context creation and candidate evaluation select a frozen
residual adapter when residual_basis is present. This adapter supplies the
stored allocation grid and verified physical entropy basis. The optimizer,
template rules, duty limits and result contracts remain shared. Returned
optimized cases preserve residual_basis and replace only utility definitions.

### C5: HPR graph selection and transport

Owners: graph analysis, graph transport schemas and presentation accessors,
with application-owned target/provenance validation.

Emit graphs only for actual matching target data. Add refrigeration GCC and
net-load graph types/methods. Accept an optional target object on HPR plot
methods, preserve graph/series metadata through serialization and resolve
selection using target provenance rather than list position. Application
validates local current/retained references; presentation does not own lifecycle
validation or run engineering analysis.

### C6: Tutorial and reporting integration

Owners: existing reporting/metric policies, notebook generator, API inventory
and Sphinx guides. Show selected/achieved duty and source context without
duplicating numerical computation. Add the public residual conversion and two
refrigeration plot methods to the supported-method inventory.

Notebook 08 demonstrates direct mode comparisons, a separate utility-placement
comparison, prices and the full residual utility workflow. Unrelated notebook
edits remain outside the correction.

## HPR correction interfaces

These signatures are proposed application contracts. Low-level function
placement may be adjusted during Code Generation without changing the public
behavior. Type definitions remain at their proper owners; the package root
continues to export only `PinchProblem` and `PinchWorkspace`.

### Public workflow

The user-selected entry point is
`problem.target.residual_utility(base_target=heat_pump)`. It belongs to the
existing target accessor and returns the detached residual case described below.

```python
class _TargetAccessor:
    def residual_utility(
        self,
        *,
        base_target: HeatPumpTargetBase,
        project_name: str | None = None,
    ) -> PinchProblem: ...
```

The target supplies its mode, zone, scalar period and snapshot. Do not add
redundant zone/period/mode selectors or infer a target from the last method run.
Require a successful eligible local result with matching provenance and result
digest. Reject foreign, stale, unavailable, weighted or unsupported aggregate
references before constructing a case. A valid retained scalar result may be
used only when the application can verify it against an authoritative retained
record without replay. A mutated result is not a source of silently recomputed
residual data.

The returned problem is initially unsolved. Creation detaches values and
validates input only. Utilities retain temperatures, prices, heat-transfer
contributions and capacity limits; inherited solved duties are reset for
allocation. Original case and HPR result remain unchanged.

```python
heat_pump = problem.target.carnot_heat_pump(
    load_fraction=0.25,
    condensers=1,
    evaporators=1,
    options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
)
load = heat_pump.hpr_load
profile = heat_pump.hpr_residual.profile

residual = problem.target.residual_utility(base_target=heat_pump)
leftover = residual.target.direct_heat_integration()
residual.plot.net_load_profiles()

optimized = residual.target.utility_placement(isothermal=2)
optimized_leftover = optimized.target.direct_heat_integration()
```

This example is proposed syntax and is not executable against the current
package. The optimized result is conditional on the frozen HPR result.

### Existing HPR calls and observations

Keep `carnot_heat_pump()` and `carnot_refrigeration()` names and load arguments.
`load_fraction` is a finite fraction within zero through one of the selected
background service duty. Explicit duty remains an alternative. The HPR result
reports available, selected and achieved duties separately through `hpr_load`.
Call/configuration precedence is unchanged; shared validation is consistent.

`hpr_residual.profile` is the precise residual representation, including any
new pocket breakpoints. Legacy problem-table projections must also be finite,
but must not truncate or replace that canonical grid.

### Public plots

```python
class PlotAccessor:
    def grand_composite_curve_with_heat_pump(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def net_load_profiles_with_heat_pump(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def grand_composite_curve_with_refrigeration(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def net_load_profiles_with_refrigeration(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...
```

`target` accepts a solved HPR target object, with period derived from that
target. Each method accepts only its matching mode. An omitted target is valid
when exactly one matching eligible target remains after zone filtering;
otherwise raise a useful unavailable/ambiguous-selection error. Multiple
periods require explicit scalar target selection. `index` remains a graph index
within the selected target, not a way to pick a different target. Reject a
conflicting zone selector. Never fall back from refrigeration to heat pumping.

Examples: `problem.plot.grand_composite_curve_with_heat_pump(target=heat_pump)`
and `problem.plot.net_load_profiles_with_refrigeration(target=refrigeration)`.
Both use cached data. Plotting a retained target is supported only when its
matching graph snapshot is available and verifiable without execution.

### Residual case supported operations

| Operation | Contract |
|---|---|
| `validate()`, `to_problem_json()`, construction from canonical input | Preserve and validate residual_basis and utility definitions. |
| `target.direct_heat_integration()` | Allocate current utilities to the frozen profile; return a residual-classified utility target. |
| `target.utility_placement()` | Optimize utility levels on the frozen residual and return another residual case. |
| `summary_frame()`, reporting, standard residual GCC/net-load plots | Read completed residual utility results and identify origin/period/basis. |
| Other engineering methods requiring original physical streams | Fail explicitly for a residual-only case unless later implemented against its typed basis. |

In particular, new heat recovery, HPR resizing, process components, HEN design
and changes to process temperature shifts cannot silently reinterpret this case.
Current-case utility changes may invalidate residual utility results, but never
change the frozen thermal basis. Reloading a complete new source follows the
ordinary problem lifecycle. Ordinary source cases retain all existing behavior.

### Owner-level interfaces

| Owner | Proposed method | Input and output |
|---|---|---|
| HPR analysis | `build_hpr_load_summary(...)` | Selected service, verified cycle/ambient/accounting result -> immutable HPRLoadSummary |
| HPR analysis | `build_hpr_residual_data(...)` | Corrected background, physical boundary, result and period -> immutable HPRResidualData |
| Application | `bind_hpr_residual_snapshot(data, provenance, result_digest)` | Verified numerical data and local provenance -> final HPRResidualSnapshot |
| Application | `resolve_hpr_target_reference(problem, target)` | Live/retained result -> validated local reference and cached snapshot |
| Application | `create_hpr_residual_case(problem, base_target, project_name=None)` | Validated reference -> detached PinchProblem |
| Residual analysis | `allocate_residual_utilities(snapshot, hot_utilities, cold_utilities, options)` | Frozen profile and utility definitions -> allocations, summary and residual graphs |
| Placement application adapter | `build_residual_placement_context(snapshot, request)` | Residual snapshot and existing request -> immutable placement context |
| Placement allocation adapter | `allocate(period, placement)` | Existing candidate protocol -> AllocationAdapterResult on the frozen basis |
| Graph application resolver | `resolve_hpr_graph_selection(problem, target, mode, zone_name)` | Explicit/unique target -> graph-set identity; no solver calls |

Detailed parameter records, result validation predicates and tolerances will be
specified in the relevant Functional Design units.

## HPR correction services and lifecycle

### S1: Calculate and publish one HPR target

1. Resolve effective load options, mode, placement, zone and selected period.
2. Compute available and selected service; run the relevant HPR model with
   separate service, ambient and feasibility accounting.
3. Verify cycle and residual consistency, including the full residual thermal
   profile and existing utility allocation. Failure must not publish a
   successful eligible residual target.
4. Construct the load summary, HPRResidualData and matching HP or refrigeration
   graph records from that verified result. Keep backend objects out of values.
5. The application binds provenance and a deterministic digest of the relevant
   result data to form the final HPRResidualSnapshot, then atomically publishes
   target, report and graph snapshots. No partially stamped snapshot is exposed.

Do not derive a second residual after reporting from rounded graph coordinates.
The numerical representation precedes and drives presentation. Different
backends share the same output contract; their equations remain backend-owned.

### S2: Observe an HPR result

Reading duty values and profiles returns immutable numerical records. Plot
selection first resolves the requested mode and target; it then retrieves that
target's cached graph set and renders or returns graph data. Graph transport
preserves name, series identity, vertical/utility flags and target/period
context. Selection errors leave the study unchanged. No observation invokes
targeting, fills a missing result by rerunning it or changes configuration.

### S3: Detach the residual

`problem.target.residual_utility(base_target=...)` validates ownership, freshness, supported
classification, scalar period, physical basis and result digest. The new input
contains residual_basis, copied utility definitions, appropriate thermal and
economic settings, and explicit selected-period/zone metadata. It contains no
original process-stream list, HEN network or process components.

Construct the new PinchProblem from this canonical input and return it unsolved.
The new case has its own analysis owner identity; the snapshot retains the
origin identity as historical provenance. Once constructed, it is independent
of the original object's future state. Repeated conversion of an unchanged
target has equal engineering content, while case-owner identities may differ.

`residual_basis` must survive all canonical input and result-case reconstruction
paths used by utility placement. A residual study cannot degrade into an empty
or original process case when cloned or serialized. A digest validates the
stored snapshot content and its claimed selection; it is an integrity check,
not a security/authentication guarantee.

### S4: Allocate utilities on the residual

Target dispatch checks the input kind before ordinary process preprocessing.
On a residual-only case, `direct_heat_integration()` invokes the residual
allocation service. That service uses the stored full-precision profile,
thermal coordinate basis and copied utility definitions, then emits a distinct
residual utility target and its GCC/net-load plots.

The selected period is the snapshot's canonical period. Omission selects that
period; a conflicting period or subzone selection fails explicitly. Reporting
keeps original plant heat recovery and frozen HPR work separate from residual
utility quantities. It must not present recomputed recovery of nonexistent
physical streams. A fully served residual yields zero utility duties without
inventing a dummy stream or temperature interval.

### S5: Optimize residual utility placement

1. Normalize utility templates, level counts, limits and options through the
   existing utility-placement request boundary.
2. Detect residual_basis and construct the residual placement context directly.
   Skip the process-only `_extract_period()` replay for this case kind.
3. Evaluate every candidate through the frozen residual allocation adapter.
   Reuse the existing optimizer and candidate result contract.
4. For thermal feasibility, use allocation coordinates with their explicit
   shift convention. For entropy/lost-work evaluation, use the separately
   preserved physical boundary with actual temperatures. Candidate utility
   temperatures are converted consistently for each use.
5. Return a detached PinchProblem containing the same residual_basis and the
   selected utility definitions, plus normal placement result evidence. Its
   subsequent allocation reproduces the winning candidate within tolerance.

The existing entropy evaluator currently expects physical composites. It may
need an internal residual-basis branch, but it must not be fed shifted GCC
temperatures under a `real_temperatures` label. The verification boundary is
utility completion conditional on fixed HPR/ambient exchange; any fixed HPR
equipment contribution must be identified separately. This is not a claim of
joint plant optimization or automatically validated whole-plant exergy.

If a target lacks a verified physical basis, reject residual utility-placement
eligibility explicitly rather than fabricating entropy or falling back to
process-only replay. Producing that basis for the supported Carnot notebook
cases is required work, not an optional fallback.

### Lifecycle and error policy

| Event | Behavior |
|---|---|
| Source changes before conversion | Reject a stale source target. |
| Source changes after conversion | Detached residual remains a valid historical study. |
| Utility definitions change in residual case | Invalidate utility results; retain frozen thermal basis. |
| Attempt to resize HPR or alter process shifts in residual case | Explicit unsupported-operation error. |
| Weighted result selected | Reject; require a supported scalar period result. |
| Target/graph record mutated or missing | Reject invalid selection; no hidden regeneration. |
| Candidate infeasible | Use existing explicit candidate diagnostics; do not alter source or frozen basis. |
| JSON input malformed or mixed with process data | Validation error before any engineering execution. |

There are no new services outside the Python process, message queues, workers
or external persistence dependencies in this design.

## HPR correction dependencies and data flow

### Dependency matrix

| Consumer | Dependency | Purpose |
|---|---|---|
| Domain HPR target models | Domain HPR value records | Typed duty/residual snapshots without outward imports |
| Input and graph contracts | Domain values and existing schema primitives | Portable residual basis and complete graph transport |
| HPR analysis | Domain values, targeting utilities and existing optimization | Calculate and verify duties/residuals |
| Application residual service | Domain values, input contracts and HPR target resolver | Validate local selection and create a detached case |
| Application target dispatch | Residual analysis service | Allocate utilities without process-stream replay |
| Placement application adapter | Residual snapshots and analysis placement context | Supply one immutable basis to every candidate |
| Placement analysis | Existing contracts, domain values and numerical helpers | Allocate/evaluate/optimize on the provided basis |
| Graph presentation | Graph contracts plus application-provided selection resolution | Select cached records and render them |
| Reporting and tutorials | Public problem/target interfaces | Explain and demonstrate verified outcomes |

Domain never imports application, analysis, presentation or contract modules.
Transport may import domain values. Analysis never imports application or
presentation. Application owns PinchProblem construction and provenance
selection; no method on a domain target constructs a problem object.

### Communication pattern

All calls are synchronous within existing library owners. Numerical snapshots
cross boundaries as finite immutable records. Live cycle/backend objects stay
inside their current execution/artifact paths. No internal serialized string
selector replaces the public named workflow methods.

### Data-flow diagram (text)

1. Original PinchProblem and explicit HPR call enter application targeting.
2. HPR analysis produces verified duty accounting, residual profiles, physical
   boundary data and graph data.
3. Application stamps and publishes the target and graph snapshots.
4. Plot observation selects and renders the matching cached graph; it stops
   there and does not lead back to analysis.
5. Explicit target.residual_utility conversion validates the selected target and creates
   a new PinchProblem input with residual_basis and utility definitions.
6. Residual utility targeting reads that basis and publishes utility results.
7. Residual utility placement supplies the same basis to every candidate,
   returns an optimized residual case, and preserves the original source.

This textual diagram is the canonical accessible representation; no graphics
renderer is required to understand the flow.

### Shared contract checkpoints

Unit 1 establishes domain load/residual values and physical correctness. Unit 2
uses target identity and verified data to repair plotting. Unit 3 adds the
canonical residual input and both dispatch/placement branches against those
values. Unit 4 integrates stable public operations into the notebook and docs.

Creation of a residual case and graph selection share a local-target resolver,
but do not share mutable caches or invoke one another. Storage/transport of
snapshots must retain tuples, finite values, units, exact profile grids and
source identity. New input semantics require validation and installed-package
checks in addition to owner-level numerical tests.
