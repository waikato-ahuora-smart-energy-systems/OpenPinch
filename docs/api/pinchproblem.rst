PinchProblem
============

:class:`OpenPinch.PinchProblem` is the canonical stateful workflow for one
process-engineering case.

Lifecycle
---------

``prepared``
   Construction or ``load(...)`` validates and prepares streams, utilities,
   zones, periods, and configuration. Input replacement is atomic: if loading,
   validation, or preparation fails, the prior prepared problem and cached
   results remain available together with their original input.

``targeted``
   A descriptive ``target`` method stores the latest result. All-period methods
   additionally populate the ordered ``period_results`` cache.

``designed``
   A descriptive ``design`` method returns a HEN design view with ranked
   network selection and grid rendering.

``invalidated``
   Loading new input, changing stored options, changing the temperature-approach
   contribution, or mutating
   a process component clears results that no longer describe the prepared
   problem.

Interaction Matrix
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 27 18 18 13

   * - Surface
     - Purpose
     - Return
     - State effect
     - Dependency
   * - ``load``, ``validate``, ``validation_report``, ``to_problem_json``
     - Prepare, check, and serialize input
     - zone, report, or mapping
     - prepare or observe
     - base
   * - ``target.direct_heat_integration``, ``indirect_heat_integration``,
       ``total_site_heat_integration``, ``all_heat_integration``
     - Core Pinch and Total Site analysis
     - target output
     - targeted
     - base
   * - ``target.heat_exchanger_area_and_cost``, ``exergy``,
       ``energy_transfer``
     - Enrich a thermal target
     - target output
     - targeted
     - base
   * - ``target.heat_recovery_dt_min``
     - Invert requested process recovery to an equivalent global ``dt_min``
     - frozen diagnostic result
     - source unchanged
     - base
   * - ``target.carnot_*``, ``vapour_compression_*``, ``brayton_*``,
       ``mvr_heat_pump``
     - Model-specific HPR studies
     - target output
     - targeted
     - HPR extras by model
   * - ``target.cogeneration`` and named turbine-model methods
     - Cogeneration screening
     - target output
     - targeted
     - base
   * - ``target.utility_placement``
     - Optimize or infer hot and cold utility levels at a selected hierarchy zone
     - detached optimized problem
     - source unchanged
     - base
   * - ``target.all_periods.*``
     - Mirror supported targeting over ordered periods
     - period-to-output mapping
     - period cache
     - method-specific

   * - ``components.add_process_mvr``, ``components.inventory``
     - Add or inspect process MVR mutations
     - component or mapping
     - invalidates on mutation
     - HPR extras
   * - ``design.*heat_exchanger_network``, ``open_hens``, ``pinch_design``,
       ``thermal_derivative``, ``network_evolution``
     - HEN synthesis and improvement
     - design view
     - designed
     - HEN solver
   * - ``summary_frame``, ``metrics``, ``report`` and state properties
     - Inspect prepared or cached state
     - dataframe, mapping, report, or record
     - none
     - base
   * - ``plot.catalog``, ``plot.data``, and named plot methods
     - Inspect cached graph data or build a figure
     - catalog, mapping, or figure
     - none
     - plotting
   * - ``plot.export``, ``plot.export_gallery``, ``export_excel``,
       ``show_dashboard``
     - Explicit publication side effects
     - paths or dashboard handle
     - none
     - output-specific

Inverse Heat-Recovery Target
----------------------------

Use the inverse target when process heat recovery is specified and the
corresponding global ``dt_min`` is unknown:

.. code-block:: python

   result = problem.target.heat_recovery_dt_min(
       heat_recovery={"value": 4_000.0, "unit": "kW"},
       zone="Site/Process",
       period_id="0",
   )

   payload = result.model_dump(mode="json")

Plain recovery values use the configured input heat-flow unit; scalar
``Value``/Pint quantities and exact ``{"value", "unit"}`` mappings are also
accepted. Input shapes are strict: Booleans, numeric strings, sequences,
arrays, and unrelated mappings are rejected. The result reports the requested and achieved
recovery, zero-``dt_min`` thermodynamic limit, residual, status, and iteration
count with explicit output units. Requests above the thermodynamic limit are
rejected with the request, limit, scope, period, and units in the error.

``HeatRecoveryDtMinResult`` is frozen and contains ``scope``, canonical
``period_id``, ``dt_min``, ``requested_heat_recovery``,
``achieved_heat_recovery``, ``thermodynamic_limit``,
``heat_recovery_residual``, ``status``, and ``iterations``. Status is
``solved`` for an interior request, ``at_thermodynamic_limit`` for the
greatest approach retaining maximum recovery, or ``zero_recovery_boundary``
for the first approach that produces zero recovery. The
``at_thermodynamic_limit`` approach may be positive for a threshold problem;
zero approach defines how the maximum recovery is calculated, not a mandatory
inverse result.

For all periods, a scalar broadcasts deliberately, while a mapping must contain
exactly the canonical period IDs. Exact period keys take precedence when the
IDs themselves are ``value`` and ``unit``:

.. code-block:: python

   period_results = (
       problem.target.all_periods.heat_recovery_dt_min(
           heat_recovery={"base": 4_000.0, "turndown": 3_200.0},
           workers=2,
       )
   )

The inverse service is non-mutating: it does not replace ordinary target
results, populate ``period_results``, alter stream contributions, or update the
last target-run specification. The value is process-level global ``dt_min``,
not exchanger-level EMAT.

Supported scopes are Site, Process Zone, and Unit Operation. Community and
Region scopes are rejected because they are not direct process-targeting
scopes. A ``Zone`` argument contributes only its address, which is resolved
against the current problem or current batch case. On an interior plateau, the returned value is the greatest feasible
approach that still meets the requested recovery.

.. autoclass:: OpenPinch.contracts.heat_recovery_dt_min.HeatRecoveryDtMinResult
   :members:
   :no-index:

The complete runnable workflow, validation matrix, unit behavior, workspace
batch examples, and numerical interpretation are in
:doc:`../guides/heat-recovery-dt-min`.

Argument Precedence
-------------------

Effective arguments resolve as ``named keyword > options > stored config >
default``. Named keywords and ``options`` apply only to that call. Use
``update_options(...)`` when a later call should inherit a persistent
engineering value. Configuration never stores which target or design method to
run.

Process MVR Component Results
-----------------------------

``components.add_process_mvr(...)`` returns the component it created. Use
engineering argument names such as ``compressor_efficiency`` and
``motor_efficiency``. The returned object exposes ``active``, ``activate()``,
``deactivate()``, ``original_streams``, ``replacement_streams``,
``stage_results_by_period``, ``affected_zone_paths``, and ``work_for_zone()``.
Changing component activity invalidates cached targets, so rerun the chosen
target method afterward.

Complete API
------------

.. autoclass:: OpenPinch.PinchProblem
   :members:
   :undoc-members:

The operation-level inventory and tutorial owner for every member is published
in :doc:`../examples/tutorial-coverage-map`.
