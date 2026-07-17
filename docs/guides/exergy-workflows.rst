Exergy Workflows
================

.. warning::

   This advanced guide uses unsupported internal owner modules. Only
   :func:`OpenPinch.main.pinch_analysis_service` is compatibility protected.

Purpose
-------

Use exergy targeting when you need exergy metrics and exergetic graphs for an
already solved thermal target.

Prerequisites
-------------

Run a compatible direct, indirect, or HPR thermal target first. Exergy is
post-processing; it enriches an existing target family rather than solving a
new thermal target from scratch.

Sample Case
-----------

Use ``pulp_mill.json`` for site-style exergy interpretation after a Total Site
target.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch.application.problem import PinchProblem

   problem = PinchProblem("pulp_mill.json")
   problem.target.indirect_heat_integration()
   exergy_target = problem.target.exergy(
       options={"base_target_type": "Total Site Target"},
   )
   summary = problem.summary_frame()

Expected Output
---------------

The selected thermal target is enriched with fields such as
``exergy_sources``, ``exergy_sinks``, ``ETE``, ``exergy_req_min``, and
``exergy_des_min``.

Interpretation
--------------

Read the exergy result after the thermal picture is clear:

1. confirm which target family was enriched
2. inspect ``exergy_sources`` and ``exergy_sinks``
3. inspect ``exergy_req_min`` and ``exergy_des_min``
4. inspect exergetic graph families

Graph accessors are available after enrichment:

.. code-block:: python

   gcc_x = problem.plot.exergetic_grand_composite_curve()
   nlp_x = problem.plot.exergetic_net_load_profiles()

Use ``zone_name=...``, ``include_subzones=True``, and ``period_id=...`` when
the exergy scope needs to match a specific thermal solve.

Next Steps
----------

- :doc:`graphing-and-interpretation` for graph reading order.
- :doc:`../api/pinchproblem` for the wrapper surface.
- :doc:`../api/service-layer` for the post-processing boundary.
