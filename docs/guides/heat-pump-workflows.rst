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
