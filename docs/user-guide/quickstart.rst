Quickstart Workflow
===================

This walkthrough shows the main Python workflow for OpenPinch: load a case,
run the analysis, inspect the summary, and export graphs and Excel results.

Step 1. Load A Known-Good Sample
--------------------------------

The packaged sample cases are the fastest way to verify that your environment
is working.

.. code-block:: python

   from pathlib import Path

   from OpenPinch import PinchProblem
   from OpenPinch.resources import copy_sample_case

   case_path = copy_sample_case("basic_pinch.json", Path("basic_pinch.json"))
   problem = PinchProblem(problem_filepath=case_path)

Step 2. Run The Analysis
------------------------

Use :meth:`OpenPinch.classes.pinch_problem.PinchProblem.run` to execute the
full targeting workflow.

.. code-block:: python

   result = problem.run()
   result.name

Step 3. Inspect The Summary
---------------------------

For quick inspection, use the compact summary table:

.. code-block:: python

   summary = problem.summary_frame()
   print(summary)

If you need the wide export-style table with value/unit columns:

.. code-block:: python

   detailed_summary = problem.summary_frame(detailed=True)

Read the compact summary in this order:

- start with ``Plant/Direct Integration`` or the main process row
- check ``Hot Utility Target`` and ``Cold Utility Target`` first
- then check ``Heat Recovery`` to see how much thermal duty is recovered internally
- use the pinch temperatures to identify the constrained temperature region

Step 4. Generate Graphs
-----------------------

Build Plotly figures directly from the solved result set:

.. code-block:: python

   gcc = problem.plot_grand_composite_curve()
   cc = problem.plot_composite_curve()

You can also export HTML graph files for later review:

.. code-block:: python

   written = problem.export_graphs("graphs", graph_type="gcc")
   print(written)

The grand composite curve is usually the first graph to inspect when you are
deciding between utility levels or considering heat-pump integration.

Step 5. Export Results
----------------------

To write the solved targets to an Excel workbook:

.. code-block:: python

   workbook_path = problem.export_excel("results")
   print(workbook_path)

Step 6. Launch The Dashboard
----------------------------

If you want an interactive dashboard after solving:

.. code-block:: python

   problem.show_dashboard()

Programmatic Payload Workflow
-----------------------------

If you prefer to construct a case directly in code, use the schema models and
service layer:

.. code-block:: python

   from OpenPinch import pinch_analysis_service
   from OpenPinch.lib.enums import StreamType
   from OpenPinch.lib.schema import StreamSchema, TargetInput, UtilitySchema

   streams = [
       StreamSchema(
           zone="Process Unit",
           name="Reboiler Vapor",
           t_supply=200.0,
           t_target=120.0,
           heat_flow=8_000.0,
           dt_cont=10.0,
           htc=1.5,
       ),
       StreamSchema(
           zone="Process Unit",
           name="Feed Preheat",
           t_supply=40.0,
           t_target=160.0,
           heat_flow=6_000.0,
           dt_cont=10.0,
           htc=1.2,
       ),
   ]

   utilities = [
       UtilitySchema(
           name="Cooling Water",
           type=StreamType.Cold,
           t_supply=25.0,
           t_target=35.0,
           heat_flow=120_000.0,
           dt_cont=5.0,
           htc=0.8,
           price=12.0,
       )
   ]

   payload = TargetInput(streams=streams, utilities=utilities)
   result = pinch_analysis_service(payload, project_name="Example")

Notebook Workflow
-----------------

OpenPinch also ships with a packaged notebook series for distinct workflows.
Copy them into your working directory with:

.. code-block:: bash

   openpinch notebook -o notebooks

The notebook series includes:

- ``01_basic_pinch_analysis.ipynb``
- ``02_graphs_and_interpretation.ipynb``
- ``03_zonal_analysis.ipynb``
- ``04_heat_pump_workflow.ipynb``
- ``05_batch_comparison.ipynb``

Next Steps
----------

- Use :doc:`../reference/api-core` for the supported API surface.
- Use :doc:`interpreting-results` for output-reading guidance.
- Use ``openpinch run`` and ``openpinch graph`` for CLI-driven workflows.
- Use the notebook series as the main learning path for distinct outputs.
