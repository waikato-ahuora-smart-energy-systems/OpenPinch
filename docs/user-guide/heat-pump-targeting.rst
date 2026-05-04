Heat-Pump Targeting
===================

This page documents the public OpenPinch workflow for evaluating a candidate
heat-pump integration scenario against a base process case.

What This Workflow Answers
--------------------------

The main question is not whether a refrigeration or heat-pump cycle can be
solved in isolation. The main question is whether a candidate temperature lift
improves the utility picture of the process.

The public workflow is designed to answer:

- which temperature lift is being tested
- which utility targets are displaced
- whether heat recovery increases
- how the grand composite curve changes after integration

Packaged Sample
---------------

Copy the packaged sample:

.. code-block:: bash

   openpinch sample --name heat_pump_targeting.json -o heat_pump_targeting.json

CLI Workflow
------------

Use the dedicated CLI command to compare a candidate scenario:

.. code-block:: bash

   openpinch heat-pump heat_pump_targeting.json \
     --condenser-temperature 170 \
     --condenser-duty 500 \
     --evaporator-temperature 90 \
     --evaporator-duty 400

This prints a comparison table showing the base case, the integrated scenario,
and the delta row.

Python Workflow
---------------

Use :meth:`OpenPinch.classes.pinch_problem.PinchProblem.evaluate_heat_pump_integration`
when you want the same workflow in code:

.. code-block:: python

   from pathlib import Path

   from OpenPinch import PinchProblem
   from OpenPinch.resources import copy_sample_case

   case_path = copy_sample_case(
       "heat_pump_targeting.json",
       Path("heat_pump_targeting.json"),
   )
   problem = PinchProblem(problem_filepath=case_path)
   evaluation = problem.evaluate_heat_pump_integration(
       {
           "zone": "Plant",
           "condenser_temperature": 170.0,
           "condenser_duty": 500.0,
           "evaporator_temperature": 90.0,
           "evaporator_duty": 400.0,
       },
       target_name="Plant/Direct Integration",
   )

   print(evaluation.comparison_frame)

Scenario Inputs
---------------

The helper accepts a small validated scenario model with these core inputs:

``zone``
   Zone that receives the integrated condenser and evaporator streams.

``condenser_temperature`` and ``condenser_duty``
   Temperature and duty of the integrated heat source delivered by the heat
   pump.

``evaporator_temperature`` and ``evaporator_duty``
   Temperature and duty of the integrated heat sink lifted from the process.

Reading The Comparison
----------------------

Start with the delta row:

- a negative ``Hot Utility Target`` delta is usually desirable
- a negative ``Cold Utility Target`` delta is also usually desirable
- a positive ``Heat Recovery`` delta indicates more internal recovery

The helper also reports an approximate heat-pump power input as the difference
between condenser duty and evaporator duty. That value is supporting context,
not the primary screening metric.

Graphs
------

After evaluating a scenario, use the returned integrated problem to inspect or
export graphs:

.. code-block:: python

   gcc = evaluation.integrated_problem.plot_grand_composite_curve(
       zone_name="Plant/Direct Integration"
   )

When the utility deltas look promising, the grand composite curve is the first
graph to inspect to understand why the scenario helped.
