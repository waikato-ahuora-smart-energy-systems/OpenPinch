Analysis State Migration
========================

The reliability changes intentionally tighten state ownership. Named analysis
methods and method-specific numerical payloads remain. No legacy aliases are
provided for direct mutation of observed state.

Observations and supported mutations
------------------------------------

``master_zone``, stream and utility collections, ``results``, ``period_ids``,
``period_results``, and placement observations are detached snapshots or
read-only mappings. Editing them does not edit a problem. Replace input through
``load``, change options through ``update_options``, and change approach
temperature multipliers through ``set_dt_cont_multiplier``. Keep component
handles returned by ``components.add_process_mvr`` and use their explicit
``activate`` and ``deactivate`` methods. The component inventory mapping cannot
be edited to register or remove a component. Input rebuilding (including
``update_options`` and ``set_dt_cont_multiplier``) continues to discard memory-only
component registrations; recreate components on the rebuilt input before solving.

Input, configuration, zone, and component changes invalidate scalar outputs,
prepared period states, graphs, and design/placement caches together. A failed
target calculation retains the preceding successful state.

Periods and prerequisites
--------------------------

Period IDs and indices always describe the calculation that produced a row.
Selected-period reports omit stale rows. A successful all-period call commits
its entire ordered batch; a failed period leaves the previous batch intact.
Worker counts must be positive integers. Workspace batches continue to isolate
failures by case.

One current prepared state per canonical period is retained in memory. For
example, ``target.all_periods.direct_heat_integration()`` followed by
``target.all_periods.exergy()`` reuses compatible thermal prerequisites. State
is replaced when superseded and is not serialized as run history. A scalar
mutation supersedes the published all-period report; run the desired all-period
workflow again to publish a complete batch.

Enrichment uses the invocation's effective settings, including exergy ambient
temperature. A thermal prerequisite produced with incompatible numerical
settings must be recalculated. A supplied ``base_target`` must belong to the
current problem, zone, period, and prepared inputs and still be current. Keep
the result returned by each enrichment if passing it to a subsequent method;
an earlier reference becomes stale when that target is replaced.

Zone objects are selectors by address. A zone observed from another problem
resolves against the receiving problem's tree. Missing and ambiguous addresses
fail before execution. With ``include_subzones=False``, child prerequisites
may be calculated privately, but child analyses are not committed or reported.

Results and provenance
-----------------------

Application-created results carry immutable ``AnalysisProvenance`` with method,
owner, zone address, period selection, input fingerprint, effective settings,
and prerequisite identities. Manually constructed low-level results may have
no application provenance and cannot be passed as current ``base_target``
references. Returned results are detached from problem-owned state.

Weighted summaries preserve existing metric policies, but no longer attach the
first period's HPR simulation record to the aggregate. Inspect the originating
period row for that evidence. Brayton remains unavailable and fails before any
analysis state is prepared or mutated.
