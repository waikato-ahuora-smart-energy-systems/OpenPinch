Heat Pump and Refrigeration Workflows
=====================================

OpenPinch exposes each HPR model as a descriptive method, so process engineers
do not need internal enum or string answers.

Carnot Screening
----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("heat_pump_targeting.json", project_name="HPR Study")
   heat_pump = problem.target.carnot_heat_pump(
       is_utility_heat_pump=False,
       is_cascade_cycle=True,
       load_fraction=0.25,
       condensers=1,
       evaporators=1,
   )
   refrigeration = problem.target.carnot_refrigeration(
       is_utility_refrigeration=True,
       load_fraction=0.25,
   )

``is_utility_heat_pump=True`` selects indirect utility placement. Cascade and
parallel topology are boolean engineering decisions. Named load values avoid a
separate load-mode string.

The Carnot methods cover Cascade Carnot cycles and parallel Carnot screening
without asking for a cycle-name string.

Load, economics and target-specific plots
-----------------------------------------

``load_fraction`` is a number from zero to one selecting a fraction of the
available net heating load (HP) or cooling load (refrigeration) near the pinch.
It is a service selection, not compressor part-load. A heat pump can supply less
than its selected ceiling if utilities are cheaper. Inspect
``heat_pump.hpr_load`` for available, selected and achieved service, cycle duties,
work and separate ambient exchanges (kW). Zero selected load returns no HPR target.
Use ``load_duty`` for an absolute duty or ``period_loads`` for per-period duties
instead of ``load_fraction``; these selectors are mutually exclusive.

Heat pumps have no penalty requiring them to consume all low-grade heat.
Refrigeration retains a feasibility term for unserved selected cooling. The
Carnot objective uses electricity-equivalent price ratios in
``COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE`` and
``COSTING_HPR_PRICE_RATIO_COLD_TO_ELE``. These differ from utility stream prices.
Both ratios default to 1.0; notebook 08 explicitly uses a positive cold ratio of
0.1 to illustrate inexpensive cooling without changing global defaults. For a
fixed candidate leaving 100 kW of cooling, this lowers the cooling-cost term
from 100 to 10 kW electricity-equivalent; it adds no heat-pump feasibility
penalty.

Use the same process/utility basis and stage counts when comparing modes.
Select a solved target explicitly when inspecting several results::

   gcc_hp = problem.plot.grand_composite_curve_with_heat_pump(target=heat_pump)
   gcc_rf = problem.plot.grand_composite_curve_with_refrigeration(target=refrigeration)
   nlp_rf = problem.plot.net_load_profiles_with_refrigeration(target=refrigeration)

Plotting only reads results. Ambiguous selections require ``target=``; an HP plot
never selects a refrigeration result. Changed or foreign target handles fail.

Residual utility allocation and placement
-----------------------------------------

Select a solved scalar HPR result, then allocate existing utilities or optimize
their temperatures against its remaining load profiles::

   utilities = problem.target.all_heat_integration(base_target=heat_pump)
   optimized = problem.target.utility_placement(base_target=heat_pump, isothermal=2)
   placement_summary = optimized.summary_frame()
   optimized_gcc = optimized.plot.grand_composite_curve()

``heat_pump`` is the solved ``DirectHeatPumpTarget`` returned by the Carnot call
above (or ``IndirectHeatPumpTarget`` for utility HPR). It retains its selected
load, cycle result, residual profiles and provenance. A zero-service call returns
``None`` and cannot be used as a basis.

The original process study and HPR result remain unchanged. ``utilities`` is a
``TargetOutput`` containing residual allocation; it does not report artificial
original-process recovery. For a single ``ResidualUtilityTarget``, use
``problem.target.direct_heat_integration(base_target=heat_pump)``.

``optimized`` is an independent, already solved ``PinchProblem``. Its summary,
plots, selected ``period_results`` and ``utility_placement_result`` are ready to
read without another targeting call. Candidate evaluation and final allocation
use the same frozen HPR profile, including physical boundary temperatures for
entropy calculations. Placement still requires at least two isothermal levels.

For an editable intermediate case, derive it explicitly::

   residual = problem.residual_utility(base_target=heat_pump)
   residual_results = residual.target.all_heat_integration()

Case derivation returns an unsolved problem. The former, unreleased
``problem.target.residual_utility`` spelling has moved to ``problem``.

Transfer the optimized utility definitions into a fresh original-process study::

   new_problem = problem.with_utilities_from(optimized)
   new_results = new_problem.target.all_heat_integration()

This copies utility definitions, not installed HPR equipment. ``new_problem``
starts unsolved, with the receiver's process streams, runtime components,
configuration and zone tree. The donor's residual and results are not copied.
Nominal segment shapes, costs, units, fluid metadata, active flags and capacity
limits are retained. Period sets must match; ordered values are aligned by ID.
Use ``project_name=`` to override the receiver's name. To keep the fixed HPR study,
continue analysing ``optimized`` or derive from an existing residual receiver.

The three HPR shortcuts accept a current, local, unmodified scalar HPR result;
strings, whole result collections and other target kinds are unsupported.
Omitted zone and period selectors inherit the HPR result. Explicit conflicts,
``include_subzones=True``, shared HPR aggregates and scalar-handle broadcasting
through all-period or workspace batches fail clearly. Thermal overrides cannot
change a frozen basis; placement search controls remain available in ``options``.

This is sequential optimization with HPR duties and temperatures fixed. To
resize HPR, retarget on a process case. Residual cases support allocation and
placement; other analyses require original physical streams. Their input JSON
retains the residual basis, but ordinary JSON serialization does not persist a
solved result cache: reconstructing it creates an unsolved case. A derived target
belongs to its derived case, rather than becoming a local handle on the source.

Simulated Models
----------------

.. code-block:: python

   # CoolProp is the default targeting backend.
   vc = problem.target.vapour_compression_heat_pump(
       refrigerants=["ammonia"],
       load_fraction=0.25,
       condensers=1,
       evaporators=1,
   )
   brayton = problem.target.brayton_heat_pump(load_fraction=0.25)
   mvr_cascade = problem.target.mvr_heat_pump(load_fraction=0.25)

Refrigeration uses ``vapour_compression_refrigeration()`` or
``brayton_refrigeration()``. Model-specific callables expose only arguments that
make sense for that model.

The named Brayton callables are part of the public workflow vocabulary, but the
current runtime raises ``NotImplementedError`` while the Brayton solver path is
being repaired. Tutorial 09 demonstrates a guarded screening pattern that reports
this limitation without interrupting the rest of a comparative study.

Set ``is_cascade_cycle=False`` for Parallel vapour compression cycles. Use the
separate ``mvr_heat_pump()`` method for Vapour compression with MVR cascade;
MVR is not hidden behind a cycle selector.

TESPy Targeting and a Target-Owned Map
--------------------------------------

Install the optional simulator only when this workflow needs it::

   python -m pip install "openpinch[tespy]"

Selecting ``simulation_backend="tespy"`` replaces the thermodynamic candidate
evaluation inside the current vapour-compression targeting method. It does not
run a second, hidden map calculation and it does not fall back to CoolProp.
CoolProp remains the result-identical default when the selector is omitted or
set explicitly to ``"coolprop"``.

The first TESPy targeting surface is one refrigerant loop with one evaporator
and one condenser. Disable features that require a different topology:

.. code-block:: python

   tespy_target = problem.target.vapour_compression_heat_pump(
       simulation_backend="tespy",
       refrigerants=["HEOS::R32[0.5]&R125[0.5]"],
       load_fraction=0.25,
       condensers=1,
       evaporators=1,
       initialize_from_carnot=False,
       allow_integrated_expander=False,
   )

The selected backend, winning temperatures, duty, working-fluid identity, and
assumptions are retained in a detached target simulation record. Generate a map
only when the downstream study needs one:

.. code-block:: python

   from OpenPinch.contracts.hpr_performance_map import (
       HprPerformanceMapRequest,
   )

   winning = tespy_target.hpr_details.target_simulation_record
   source_at_design = (
       winning.nominal_evaporating_temperature
       + winning.source_approach_temperature
   )
   sink_at_design = (
       winning.nominal_condensing_temperature
       - winning.sink_approach_temperature
   )
   request = HprPerformanceMapRequest(
       map_id="selected-heat-pump",
       source_temperatures=[source_at_design],
       sink_temperatures=[sink_at_design],
       load_fractions=[0.50, 0.75, 1.00],
   )
   performance_map = problem.target.hpr_performance_map(
       target=tespy_target,
       request=request,
   )
   plain_json_data = performance_map.model_dump(mode="json")

The map backend comes from ``tespy_target`` rather than current configuration.
An explicit ``reference_capacity`` on the request overrides the winning useful
duty; otherwise that duty is the fixed reference capacity. The returned schema
contains active operating points only. OpenUtility or another MILP package can
consume ``plain_json_data`` without importing OpenPinch.

Notebook 09 is the comprehensive executable target-to-map example. It starts
with the omitted/default CoolProp selector, inspects the detached winning
record, derives the design source and sink coordinates, generates three
part-load points, and serializes the versioned result as plain JSON. A separate
guarded call demonstrates explicit TESPy selection and molar-mixture syntax;
failure remains a TESPy screening result and never triggers CoolProp fallback.

Working-fluid strings accept installed-property pure fluids, registered blends,
and explicit N-component molar mixtures. For example,
``HEOS::R32[0.3]&R125[0.3]&R143a[0.4]`` retains component order and normalized
molar composition. Support is determined at the required operating states, not
by an OpenPinch refrigerant allowlist; ``REFPROP`` specifications are rejected.

Multiperiod HPR
---------------

.. code-block:: python

   periods = PinchProblem("crude_preheat_train_multiperiod.json")
   outputs = periods.target.all_periods.carnot_heat_pump(load_fraction=0.25)
   weighted = periods.summary_frame(include_weighted_average=True)

TESPy supports one selected scalar period and independent
``target.all_periods.vapour_compression_heat_pump(...)`` replay with
``workers=1``. A shared-vector multiperiod TESPy optimization and automatic map
generation across periods are not supported. Run independent processes if
separate calls must execute in parallel; external thermodynamic engines have no
thread-safety promise here.

Process MVR
-----------

.. code-block:: python

   process = PinchProblem("process_mvr.json", project_name="Site")
   component = process.components.add_process_mvr(
       "Evaporator vapour",
       liquid_injection=False,
       compressor_efficiency=0.72,
       motor_efficiency=0.96,
   )
   target = process.target.direct_heat_integration()
   inventory = process.components.inventory
   work = component.work_for_zone(process.master_zone)
   replacements = component.replacement_streams

Adding, activating, or deactivating a component invalidates target results.
Run the desired target method again before reading summaries or plots.

See notebooks 08 through 11 in :doc:`../examples/notebook-series`.

The corresponding packaged files are
``08_carnot_heat_pump_and_refrigeration.ipynb``,
``09_vapour_compression_and_brayton.ipynb``,
``10_multiperiod_heat_pumps.ipynb``, and
``11_process_mvr_and_cascade.ipynb``.
