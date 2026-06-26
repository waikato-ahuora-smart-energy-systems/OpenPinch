Exergy Workflows
================

OpenPinch exposes exergy targeting as an advanced post-processing workflow on
top of an already solved thermal target.

Question This Guide Answers
---------------------------

How do I enrich an existing OpenPinch thermal target with exergy metrics and
exergetic graph views?

Typical Workflow
----------------

1. solve the base thermal case
2. choose the target family whose GCC or net-load view you want to interpret
3. run the exergy workflow on that existing result
4. inspect the exergy summary fields and exergetic graph data together

Python Surface
--------------

The main user-facing route is:

.. code-block:: python

   problem = PinchProblem("pulp_mill.json")
   problem.target()
   exergy_target = problem.target.exergy()

``problem.target.exergy()`` does not create a separate target family. It returns
the selected existing target after enriching it with:

- ``exergy_sources``
- ``exergy_sinks``
- ``ETE``
- ``exergy_req_min``
- ``exergy_des_min``

Base Target Selection
---------------------

By default, ``problem.target.exergy()`` resolves the first compatible existing
target family in this order:
``Total Site -> Indirect Heat Pump -> Direct Heat Pump -> Direct Integration``.

To pin one exact family and disable fallback, pass
``options={"base_target_type": "..."}``.

Unlike the main thermal targeting accessors, the exergy workflow expects that
the requested target family already exists for the selected zone and period.
If it does not, run the corresponding base targeting accessor first.

Typical explicit pattern:

.. code-block:: python

   problem.target.indirect_heat_integration(period_id="peak")
   ts_exergy = problem.target.exergy(
       period_id="peak",
       options={"base_target_type": "Total Site Target"},
   )

Graphs
------

The exergy workflow also restores two graph accessors:

- ``problem.plot.exergetic_grand_composite_curve(...)``
- ``problem.plot.exergetic_net_load_profiles(...)``

Example:

.. code-block:: python

   problem.target.direct_heat_integration()
   exergy_target = problem.target.exergy(
       options={"base_target_type": "Direct Integration"},
   )
   gcc_x = problem.plot.exergetic_grand_composite_curve()
   nlp_x = problem.plot.exergetic_net_load_profiles()

Interpretation Notes
--------------------

Read the exergy result after the thermal picture is already understood.

Use this order:

1. confirm which thermal target family was enriched
2. inspect ``exergy_sources`` and ``exergy_sinks``
3. inspect ``exergy_req_min`` and ``exergy_des_min``
4. inspect the exergetic GCC and exergetic net-load profiles

Zone and State Scope
--------------------

The same high-level targeting controls are available here:

- ``zone_name=...`` for one zone in the hierarchy
- ``include_subzones=True`` to enrich a selected subtree
- ``period_id=...`` for one canonical period

When ``include_subzones=True`` is used, exergy targeting is applied in
post-order so child zones are solved before any site-level exergy enrichment
that depends on their existing targets.

Next Steps
----------

- For graph reading guidance, see :doc:`graphing-and-interpretation`.
- For the exact wrapper surface, see :doc:`../api/pinchproblem`.
- For the lower-level service boundary, see :doc:`../api/service-layer`.
