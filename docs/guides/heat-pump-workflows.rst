Heat Pump Workflows
===================

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
notebook 08 when the question is direct process gas/vapour recompression.

Runnable Workflow
-----------------

Direct or indirect HPR targeting:

.. code-block:: python

   from OpenPinch import PinchProblem
   from OpenPinch.lib.enums import HPRcycle

   problem = PinchProblem("chocolate_factory.json")
   problem.update_options({"HPR_TYPE": HPRcycle.CascadeCarnot.value})
   base = problem.target.direct_heat_integration()
   hpr = problem.target.direct_heat_pump()
   site_hpr = problem.target.indirect_heat_pump()
   summary = problem.summary_frame()

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
component work in later target summaries.

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

- ``03_carnot_hpr_comparison.ipynb`` for direct/indirect HPR comparison.
- ``07_vapour_compression_mvr_cascade_hpr.ipynb`` for the VC+MVR cascade
  backend.
- ``08_direct_gas_stream_mvr.ipynb`` for process-component MVR scenarios.

Copy them with:

.. code-block:: bash

   openpinch notebook --name 03_carnot_hpr_comparison.ipynb -o notebooks

Next Steps
----------

- :doc:`../fundamentals/heat-pump-and-refrigeration-methods` for cycle
  conventions and backend details.
- :doc:`graphing-and-interpretation` for reading HPR graph effects.
- :doc:`../api/pinchproblem` for the public accessors.
