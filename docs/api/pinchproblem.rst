PinchProblem
============

:class:`OpenPinch.PinchProblem` is the canonical stateful workflow for one
process-engineering case.

Lifecycle
---------

``prepared``
   Construction or ``load(...)`` validates and prepares streams, utilities,
   zones, periods, and configuration.

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
