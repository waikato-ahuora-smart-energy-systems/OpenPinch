# HPR derived-case API requirements

Status: proposal consolidated for the requested planning task. Runtime changes
are not implemented. This extends the completed HPR notebook correction.

## Intent and scope

Make ordinary allocation, utility placement and case derivation intuitive after
a selected heat-pump or refrigeration target. Keep existing result types and
numerical algorithms, remove unnecessary user orchestration, and preserve the
frozen HPR basis. Scope is one Python library across application coordination,
contracts, public inventories, notebooks and documentation. Risk is moderate,
with particular attention to ownership, period handling and result publication.

The prior discussion establishes the preferred interface. Routine design choices
below are recommendations for implementation; they do not require another
questionnaire to produce the requested plan. Existing PBT enablement and disabled
Security/Resiliency choices persist.

## First-delivery requirements

| ID | Requirement |
|---|---|
| API-01 | Move case creation to problem.residual_utility(base_target=heat_pump); return an unsolved detached PinchProblem. |
| API-02 | Add optional base_target to direct_heat_integration, all_heat_integration and utility_placement. Initially support compatible scalar HPR results for the new residual shortcuts. |
| API-03 | With an HPR base target, direct/all integration allocate the existing utility definitions against its exact frozen residual. No HPR solve or original process/site recovery recalculation occurs. |
| API-04 | Preserve return types: direct allocation returns ResidualUtilityTarget; all allocation returns TargetOutput; placement returns PinchProblem. Label residual results explicitly. |
| API-05 | Placement returns a fully allocated case with cached summaries, appropriate plots and placement evidence. Apply this to ordinary and residual placement. |
| API-06 | Add problem.with_utilities_from(other_problem, project_name=None), returning a fresh unsolved case using the receiver's process inputs and the donor's canonical utility definitions. |
| API-07 | Case derivation and the new HPR shortcuts preserve source inputs, source solved results and the selected HPR record. Preserve exact thermal basis, period identity and provenance. |
| API-08 | Share validation rules with existing exergy, energy_transfer and cogeneration base_target consumers, preserving their established successful target combinations and publication behavior. |
| API-09 | Infer omitted scope/period from an explicit scalar target; reject contradictory selectors, unsupported target types, modified/foreign/stale handles, incompatible thermal overrides and implicit aggregate expansion. |
| API-10 | Preserve utility units, temperatures, segmented/profile shapes, active flags, prices, heat-transfer data, fluid metadata and maximum-duty constraints during transfer. Allocated duties and cached analysis are not configuration to transplant. |
| API-11 | Preserve no-base targeting behavior and existing solver objectives. Only placement's return readiness changes for ordinary callers. Migrate the newly introduced target.residual_utility spelling to its problem-level owner. |
| API-12 | Document types and fixed-HPR versus original-process workflows, migrate examples without overwriting independent notebook edits, and verify public catalogs and workspace/all-period forwarding. |

## Scope boundaries

A base_target selects a solved state; it is not a generic optimizer warm start.
What data the consumer uses depends on the operation: allocation uses the residual,
while exergy and other assessments need their own compatible physical data.

This delivery does not broaden all target methods indiscriminately. Area/cost and
implemented HPR methods are follow-on candidates requiring separate capability
contracts. Indirect/total-site integration needs aggregate/zone-hierarchy data;
a scalar HPR residual is insufficient. Keep heat_recovery_dt_min based on original
streams and hpr_performance_map's explicit required target argument. Brayton
availability, joint HPR/utility optimization and embedding targeted HPR as process
equipment are unchanged.

## Acceptance scenarios

1. A process engineer targets HP or RF, then allocates residual utilities in one
   call and obtains the same duties/profiles as explicit residual-case allocation.
2. An engineer optimizes residual utilities in one call and immediately reads
   summary/plots; HPR settings and source results have not changed.
3. An engineer creates a residual case explicitly to edit utilities before
   allocation or placement, with the same numerical behavior as the shortcut.
4. An engineer transfers the optimized utility definitions to the original
   process case and receives an independent unsolved study, without implicitly
   embedding HPR or carrying over residual analysis results.
5. An engineer runs ordinary placement on a selected zone/period set and receives
   complete results for exactly that selection, with placement evidence retained.
6. Invalid ownership, period, scope or capability combinations fail before an
   unrelated analysis runs; failures do not partly mutate either input study.

## Non-functional requirements

No new runtime dependency, infrastructure, deployment or solver model is needed.
Keep layer boundaries and root exports intact. Avoid repeated optimization or
HPR solves; one final deterministic allocation replay is acceptable to populate
canonical results. Preserve exception atomicity and bounded search options.
Do not infer serialized state from cached summaries. Keep the existing CI gate:
branch-enabled combined line/branch coverage of at least 95 percent.

## Extension applicability

PBT-01: properties and state transitions are specified in the execution plan.
PBT-02 through PBT-07 and PBT-10: planned construction obligations, not claimed
implemented. PBT-08/09: existing Hypothesis, shrinking and CI seed 20260715 retained.
Security and Resiliency: disabled and skipped. No blocking planning finding.
