Adding Analysis Methods
=======================

Keep the public API descriptive: add a named method to ``problem.target`` or
``problem.design``. The internal catalog describes capabilities; it is not a
runtime plugin registry or a public string dispatcher.

This page covers ``problem.target`` methods. A ``problem.design`` method
follows the same steps with these differences: its catalog name starts with
``design.`` and uses ``adapter="hen"``; its accessor method is in
``application/_problem/accessors/design.py`` and calls
``_DesignAccessor._run``, which snapshots the problem, stamps provenance and
commits the result, instead of using ``@target_transaction``; and the
calculation lives under ``OpenPinch/analysis/heat_exchanger_networks/``
(HEN is heat exchanger network).

All paths below are relative to ``OpenPinch/`` unless they start with
``tests/``, ``docs/`` or ``scripts/``. The worked example is
``target.heat_recovery_dt_min``, which finds the global ``dt_min`` that gives a
requested amount of process heat recovery. It is a nonthermal, non-mutating
method, so the steps that apply only to thermal targets say so.

Contribution checklist
----------------------

1. **Choose the family.** A thermal method adds a target to the problem and
   reports through the thermal serializer (``adapter="thermal"``, a target
   class in ``domain/targets.py``). A nonthermal method returns its own frozen
   contract and has its own report adapter; it needs no utility metrics and no
   ``UtilitySummaryTarget``. ``heat_recovery_dt_min`` is nonthermal.
2. **Write the failing test first.** Follow ``AGENTS.md``: a failing test,
   then the minimum implementation, then refactor on green. The test and the
   implementation go in the same commit. Tests mirror the package under
   ``tests/``. ``heat_recovery_dt_min`` has
   ``tests/analysis/test_heat_recovery_dt_min.py`` (calculation),
   ``tests/application/test_heat_recovery_dt_min.py`` (service),
   ``tests/application/test_heat_recovery_dt_min_api.py`` (signatures and
   workspace delegation) and ``tests/contracts/test_heat_recovery_dt_min.py``
   (contract). Cover these categories, mirroring the existing tests:

   - Property tests use Hypothesis with seed ``20260715`` and keep shrinking,
     as in ``tests/application/test_heat_recovery_dt_min_properties.py``, which
     also checks that parallel workers match a serial run.
   - Non-mutation and snapshot isolation:
     ``test_selected_and_all_period_calls_are_non_mutating`` in
     ``tests/application/test_heat_recovery_dt_min.py``. For a thermal method,
     also exercise a failure after a prior success and check that the complete
     previous state survives.
   - JSON round trip:
     ``test_result_is_frozen_strict_finite_and_json_round_trippable`` in
     ``tests/contracts/test_heat_recovery_dt_min.py``.
   - Provenance and report adapter:
     ``test_nonthermal_result_has_family_report_adapter`` in
     ``tests/application/test_analysis_contracts.py``.
   - Command layer: a ``run_command`` case in
     ``tests/application/test_commands.py``. No registration is needed;
     ``application/commands.py`` resolves ``target.<name>`` by attribute lookup
     on the accessor.

3. **Put the calculation in ``analysis/``.** It is deterministic and imports
   only the owners in :doc:`architecture`. For the example,
   ``analysis/targeting/heat_recovery_dt_min.py`` holds
   ``solve_heat_recovery_dt_min``.
4. **Add the catalog entry.** In ``application/_problem/targeting/catalog.py``
   add one line to ``METHOD_CATALOG``, keyed ``"target.<name>"``, built with
   ``_spec(...)``. The example is::

      "target.heat_recovery_dt_min": _spec(
          periods="nonmutating", adapter="inverse_dt_min"
      ),

   ``scopes`` are ``ZoneType`` letters (``R`` region, ``C`` community, ``S``
   site, ``P`` process zone, ``UO`` unit operation, from ``domain/enums.py``);
   the defaults are the ``PROCESS_SCOPES`` and ``AGGREGATE_SCOPES`` constants
   in the same file. ``prerequisites`` names targets that must exist first
   (for example ``("direct_heat_integration",)``). ``periods`` is one of
   ``independent``, ``nonmutating``, ``shared``, ``specialized`` or
   ``unsupported``. ``dependencies`` lists optional extras or external
   solvers, and ``available=False`` makes ``require_available`` raise
   ``NotImplementedError``. Every spec lists all user configuration fields
   (``USER_CONFIG_FIELD_SPECS``) as its configuration; ``_spec`` has no
   argument to narrow them. Unsupported methods fail in
   ``require_available`` before any input state is prepared.
5. **Define the result contract and provenance.** Put the result model in
   ``contracts/``: ``contracts/heat_recovery_dt_min.py`` defines the frozen
   ``HeatRecoveryDtMinResult`` with a ``provenance: AnalysisProvenance | None``
   field. ``AnalysisProvenance`` (``domain/analysis.py``) carries the shared
   metadata; keep method-specific payload in the owning contract. Add the
   contract to ``CONTRACTS`` in ``contracts/schema.py`` so it appears in the
   JSON schemas, and add a unit rule for each new metric name to
   ``OUTPUT_UNIT_RULES`` in ``contracts/units.py``. For a thermal method,
   subclass the target class in ``domain/targets.py`` and override
   ``prerequisite_types()`` for its prerequisite target types and
   ``provenance_settings()`` for calculation-affecting runtime choices outside
   configuration. Heat pump and refrigeration (HPR) targets include their
   effective simulation backend this way, so switching engines invalidates
   superseded references.
6. **Write the application service.** For the example,
   ``calculate_heat_recovery_dt_min`` and
   ``calculate_all_period_heat_recovery_dt_min`` in
   ``application/heat_recovery_dt_min.py`` call the calculation, convert units,
   and attach provenance with ``make_provenance`` from
   ``application/_problem/targeting/provenance.py``. A thermal or enrichment
   method instead decorates its accessor method with ``@target_transaction``
   (``application/_problem/targeting/state.py``). The decorator runs the method
   on an isolated copy of the problem (the scratch state), calls
   ``require_available``, and commits only on success, so a failure leaves the
   previous results in place. Snapshot the zone and components together; do not
   rebuild runtime state from problem JSON, because process MVR replacement
   streams and compressor records exist only at runtime. Reuse a prerequisite
   only when its period and numerical settings match, and report an actionable
   error when one is missing. Child calculations used only as prerequisites
   stay in scratch state.
7. **Expose the method on the accessors.** The example has explicit methods
   because their signatures are keyword-only:
   ``_TargetAccessor.heat_recovery_dt_min`` and
   ``_AllPeriodsTargetAccessor.heat_recovery_dt_min`` in
   ``application/_problem/accessors/target.py``, and
   ``_CaseBatchTargetAccessor.heat_recovery_dt_min`` and
   ``_CaseBatchAllPeriodsTargetAccessor.heat_recovery_dt_min`` in
   ``application/workspace.py``. ``PinchWorkspace`` has its own active-case
   method with the same name. ``_TargetAccessor`` methods are always written
   out. For the all-period and workspace batch accessors,
   ``install_catalog_forwarders`` adds a ``**kwargs`` forwarder for every
   catalog name that the class does not define, so a method with a plain
   signature needs no code there. Tests fail when the catalog and these
   accessors disagree on the set of method names.
8. **Add the report adapter.** Adapters are keyed by result class and looked up
   along the class hierarchy by ``adapt_analysis_result`` in
   ``presentation/reporting/adapters.py``. Thermal results are in
   ``presentation/reporting/thermal.py`` (``THERMAL_RESULT_ADAPTERS``); the
   example is listed in ``SPECIALIST_RESULT_ADAPTERS`` in
   ``presentation/reporting/specialist.py``. Presentation may import only
   ``adapters``, ``analysis``, ``contracts``, ``domain`` and ``presentation``,
   not ``application`` or ``optimisation``
   (``tests/architecture/test_dependency_rules.py``). Keep reports, plots and
   exports observational.
9. **Declare report fields (thermal results only).** In
   ``contracts/report_metrics.py``, ``report_field(name, policy, ...)`` gives
   every field of a report model its unit, representation and aggregation
   policy. ``metric_specifications`` raises for an undeclared field, and a
   derived field needs an implementation in
   ``application/_problem/periods/aggregation.py``, which raises
   ``ValueError`` for one that has none. See Aggregation contracts.
   ``HeatRecoveryDtMinResult`` is not a report row and declares none.
10. **Update documentation.** Edit by hand:
    ``docs/overview/capability-matrix.rst``, the method tables in
    ``docs/api/pinchproblem.rst`` and ``docs/api/pinchworkspace.rst``,
    ``docs/overview/analysis-migration.rst`` when behavior of existing state
    changes, and an entry under "Unreleased" in ``docs/release-notes.rst``.
    Add the method to ``TARGET_TUTORIALS`` (and ``ALL_PERIOD_TUTORIALS`` when it
    has an all-period mirror), or to ``DESIGN_TUTORIALS`` for a ``problem.design``
    method, in ``scripts/generate_tutorial_coverage.py``,
    mapping it to a notebook number under ``OpenPinch/tutorials/notebooks``;
    the generator raises ``KeyError`` for an unmapped public method. Then
    regenerate and commit the two generated files, which tests compare with the
    live catalog:

    .. code-block:: bash

       uv run --no-sync python scripts/generate_analysis_catalog.py
       uv run --no-sync python scripts/generate_tutorial_coverage.py

    These write ``docs/_data/analysis-methods.csv`` and
    ``docs/_data/tutorial-coverage.csv``.
11. **Verify.** Run ``scripts/ci_local.sh all`` (lint, unit, docs, TESPy,
    performance, solver, notebooks and build lanes). It does not run the
    Pyodide lane, the ``floors`` lane (runtime dependencies at their lowest
    supported versions), the py312 and py313 unit lanes, the non-core and
    non-Linux smoke tests, or the dependency audit; CI runs those. See
    :doc:`build-and-coverage`.

Adapter names
-------------

The catalog ``adapter`` string (``thermal``, ``inverse_dt_min``, ``load_kinks``,
``placement``, ``hpr_map``, ``hen``, ``residual``) is a label. It appears in
``docs/_data/analysis-methods.csv`` and nothing at runtime resolves it;
reports are chosen by result class as in step 8.
``test_catalog_workspace_period_adapters_and_docs_remain_consistent`` in
``tests/application/test_analysis_contracts.py`` fails when a spec uses a name
outside that list, so a new family name is added to the test as well.

Existing nonthermal example
---------------------------

``target.heat_recovery_dt_min`` demonstrates the path without extending the
thermal serializer. Its application service performs an isolated inverse
calculation, returns ``contracts.heat_recovery_dt_min.HeatRecoveryDtMinResult``
with immutable provenance, and uses the specialist reporting adapter. The
result retains inverse-specific feasibility and thermodynamic-limit evidence;
it does not acquire ``Qh`` or other unrelated thermal report fields.

Aggregation contracts
---------------------

``contracts/report_metrics.py`` validates the policy on every report field;
``application/_problem/periods/aggregation.py`` applies it to build the
weighted row. The policies (``AggregationPolicy``) are:

- ``weighted_mean``: the period-weighted mean of the period values.
- ``maximum``: the largest period value.
- ``consensus``: the common value when every non-null period agrees, and
  ``None`` otherwise.
- ``derived``: recomputed from other aggregated fields, such as ratios of
  totals and utility summaries; each needs an implemented derivation.
- ``exclude``: omitted from a weighted row and kept on its period rows.

The existing thermal payload weights operating metrics, takes the maximum of
HPR capital requirements, retains consensus metadata, and derives totals and
utility summaries. Period-specific HPR simulation records are excluded from a
weighted row and remain on their originating period rows. An aggregate is not
a simulated operating point.
