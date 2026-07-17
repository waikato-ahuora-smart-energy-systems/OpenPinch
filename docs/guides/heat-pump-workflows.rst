Heat Pump Workflows
===================

.. warning::

   Heat-pump owner modules and parent accessors are unsupported internals. Only
   :func:`OpenPinch.main.pinch_analysis_service` is compatibility protected.

Purpose
-------

Use this guide when you need to screen Heat Pump, refrigeration, or direct
gas/vapour MVR opportunities in the context of an OpenPinch thermal target.

Prerequisites
-------------

Run and interpret a base direct or indirect integration target first. Install
``openpinch[notebook]`` when you want graph rendering or the packaged
notebooks.

Sample Case
-----------

Use ``chocolate_factory.json`` for direct-versus-indirect HPR comparison and
``heat_pump_targeting.json`` for a compact direct screening input. Use
``crude_preheat_train_multiperiod.json`` when one HPR design must serve
several weighted operating periods. Use notebook 05 when the question is
direct process gas/vapour recompression.

Runnable Workflow
-----------------

Direct or indirect HPR targeting:

.. code-block:: python

   from OpenPinch.application.problem import PinchProblem
   from OpenPinch.domain.enums import HPRcycle

   problem = PinchProblem("chocolate_factory.json")
   problem.update_options({"HPR_TYPE": HPRcycle.CascadeCarnot.value})
   base = problem.target.direct_heat_integration()
   hpr = problem.target.direct_heat_pump()
   site_hpr = problem.target.indirect_heat_pump()
   summary = problem.summary_frame()

Opt in to one shared HPR design across named operating periods:

.. code-block:: python

   multiperiod = PinchProblem("crude_preheat_train_multiperiod.json")
   multiperiod.update_options(
       {
           "HPR_TYPE": HPRcycle.CascadeCarnot.value,
           "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True,
       }
   )
   shared_hpr = multiperiod.target.direct_heat_pump(period_id="base")
   weighted_summary = multiperiod.summary_frame(periods="weighted_average")

Refrigeration uses the companion accessors:

.. code-block:: python

   direct_refrigeration = problem.target.direct_refrigeration()
   indirect_refrigeration = problem.target.indirect_refrigeration()

Direct process MVR mutates a prepared problem through a process component:

.. code-block:: python

   component = problem.add_component.process_mvr(...)
   rerun_target = problem.target.direct_heat_integration()

Expected Output
---------------

HPR and refrigeration targeting add target rows with HPR cost, duty, COP, and
graph effects. Direct gas/vapour MVR adds replacement hot streams and includes
component work in later target summaries. Multiperiod HPR shared-design mode
keeps the requested period target row and stores all-period evaluations on
``hpr_details`` for weighted summary reporting. Candidate designs are ranked by
weighted operating cost and feasibility penalty plus the largest period's
annualized capital cost, so equipment is sized for the peak-capital period even
when another period dominates operating cost.

Weighted HPR summary rows average operating quantities and operating cost, use
the maximum total, annualized, compressor, and heat-exchanger capital fields,
and recompute total annualized cost as weighted operating cost plus maximum
annualized capital. Other target fields retain the normal weighted-average
policy. Summary replay uses isolated copies and leaves the selected problem zone
and cached result unchanged, including when a replayed period fails.

Interpretation
--------------

Compare HPR results in this order:

1. hot utility target change
2. cold utility target change
3. heat recovery change
4. total annualized HPR cost for simulated-cycle backends
5. GCC or net-load profile changes

Start broad screening with ``HPR_TYPE = "Cascade Carnot cycles"`` or
``"Parallel Carnot cycles"``. Move to ``"Parallel vapour compression cycles"``,
``"Cascade vapour compression cycles"``, or
``"Vapour compression with MVR cascade"`` only when refrigerant-specific
behavior matters.

Recommended Learning Assets
---------------------------

- ``04_carnot_heat_pump_screening.ipynb`` for direct/indirect HPR comparison.
- ``05_direct_gas_stream_mvr_scenarios.ipynb`` for process-component MVR scenarios.
- ``06_vapour_compression_mvr_cascade_hpr.ipynb`` for the VC+MVR cascade
  backend.
- ``10_multiperiod_hpr_shared_design.ipynb`` for shared HPR design across
  weighted operating periods.

Copy them with:

.. code-block:: bash

   openpinch notebook --name 04_carnot_heat_pump_screening.ipynb -o notebooks

Next Steps
----------

- :doc:`../fundamentals/heat-pump-and-refrigeration-methods` for cycle
  conventions and backend details.
- :doc:`graphing-and-interpretation` for reading HPR graph effects.
- :doc:`../api/pinchproblem` for the internal parent accessors.
