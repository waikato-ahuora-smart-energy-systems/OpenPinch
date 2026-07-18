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

   vc = problem.target.vapour_compression_heat_pump(
       refrigerants=["water", "ammonia"],
       load_fraction=0.25,
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

Multiperiod HPR
---------------

.. code-block:: python

   periods = PinchProblem("crude_preheat_train_multiperiod.json")
   outputs = periods.target.all_periods.carnot_heat_pump(load_fraction=0.25)
   weighted = periods.summary_frame(include_weighted_average=True)

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
