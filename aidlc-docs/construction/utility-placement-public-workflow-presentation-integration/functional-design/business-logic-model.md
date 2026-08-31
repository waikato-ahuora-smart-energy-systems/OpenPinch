# Unit 3 Business Logic Model

## Public problem workflow

1. Normalize explicit keyword arguments into one frozen request before target execution.
2. Resolve automatic scope to direct integration unless Total Site is explicitly requested.
3. Resolve the requested zone and canonical ordered periods.
4. Reconstruct each base target in a fresh `PinchProblem` made from the source problem JSON.
5. Extract immutable shifted and real profiles, residual demands, entropy slices, physical coordinate bounds, period weights, ambient temperature, and turbine settings.
6. Delegate once to the Unit 2 service for one shared placement vector across all selected periods.
7. Store only the detached result in a dedicated observation slot and return it.

The source problem's zone, legacy target result, selected period, and target-run
metadata are unchanged on success and failure.

## Scope and period flow

`AUTO` resolves to `DIRECT`. `DIRECT` executes existing direct heat-integration
targeting on an isolated problem. `TOTAL_SITE` executes the existing Total Site
workflow on an isolated problem. Omitted periods mean the canonical period
sequence; the explicit all-period accessor calls the same shared-vector method
and never enters the independent-period loop.

## Observation and batch flow

The problem exposes the last detached placement result without hidden work.
Metrics, summary frames, and reports consume an explicit or cached result only.
Workspace case batches reuse `CaseBatchResult`, retain case order, isolate typed
failures, and preserve the active case.

## Notebook flow

The generator emits exactly `19_utility_placement_optimisation.ipynb`. It loads
a packaged example, executes default thermodynamic placement, executes monetary
placement with priced and cogeneration-eligible templates, and asserts the
corresponding decomposition fields. It contains no CLI call.

## PBT-01 property inventory

- Request/result JSON round trips survive the public workflow.
- Repeated identical public calls are equal for a fixed seed.
- Source serialization and existing target observation are invariant on success and failure.
- Explicit period order and workspace case order are preserved.
- One case failure does not remove another case success.
- Observation methods are pure and never invoke targeting or optimisation.
