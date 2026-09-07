# HPR correction components

## C1: HPR result and residual values

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

## C2: HPR numerical service

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

## C3: Residual case application service

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

## C4: Residual utility evaluation

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

## C5: HPR graph selection and transport

Owners: graph analysis, graph transport schemas and presentation accessors,
with application-owned target/provenance validation.

Emit graphs only for actual matching target data. Add refrigeration GCC and
net-load graph types/methods. Accept an optional target object on HPR plot
methods, preserve graph/series metadata through serialization and resolve
selection using target provenance rather than list position. Application
validates local current/retained references; presentation does not own lifecycle
validation or run engineering analysis.

## C6: Tutorial and reporting integration

Owners: existing reporting/metric policies, notebook generator, API inventory
and Sphinx guides. Show selected/achieved duty and source context without
duplicating numerical computation. Add the public residual conversion and two
refrigeration plot methods to the supported-method inventory.

Notebook 08 demonstrates direct mode comparisons, a separate utility-placement
comparison, prices and the full residual utility workflow. Unrelated notebook
edits remain outside the correction.
