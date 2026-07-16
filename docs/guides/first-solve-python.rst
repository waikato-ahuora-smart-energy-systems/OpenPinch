First Solve with Python
=======================

Purpose
-------

Use :class:`OpenPinch.PinchProblem` when you want one object to load, validate,
solve, summarize, graph, export, and rerun one OpenPinch case. This is the
canonical beginner workflow and the default path for scripts and notebooks.

Prerequisites
-------------

Install the base package:

.. code-block:: bash

   python -m pip install openpinch

Install the notebook extra when you want Plotly graphs, Jupyter notebooks, or
Excel import/export:

.. code-block:: bash

   python -m pip install "openpinch[notebook]"

Sample Case
-----------

Use ``basic_pinch.json`` first. If no local file with that name exists,
``PinchProblem`` loads the packaged sample case from the installed wheel.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="basic_pinch")

   validation = problem.validation_report()
   result = problem.target()
   summary = problem.summary_frame()

   print(validation.valid)
   print(summary[["Target", "Hot Utility Target", "Cold Utility Target"]])

Expected Output
---------------

The solve caches a structured result on ``problem`` and returns a summary table
with target rows. For the first pass, inspect:

1. ``Hot Utility Target``
2. ``Cold Utility Target``
3. ``Heat Recovery``
4. ``Hot Pinch`` and ``Cold Pinch``

Graph and Export Workflow
-------------------------

With ``openpinch[notebook]`` or ``openpinch[dashboard]`` installed:

.. code-block:: python

   gcc = problem.plot.grand_composite_curve()
   catalog = problem.plot.catalog()
   workbook_path = problem.export_excel("results")
   graph_paths = problem.plot.export("graphs", graph_type="gcc")

The Grand Composite Curve is the best first graph for utility placement and
Heat Pump opportunity screening.

Period-Valued Cases
-------------------

Named targeting accessors accept ``period_id=...`` when the case input carries
multiperiod values:

.. code-block:: python

   multiperiod_problem = PinchProblem(
       "crude_preheat_train_multiperiod.json",
       project_name="crude_multiperiod",
   )

   print(multiperiod_problem.period_ids)
   peak_target = multiperiod_problem.target.direct_heat_integration(period_id="peak")
   all_period_results = multiperiod_problem.target_all_periods(parallel="thread")

The summary, export, and graph surfaces then reflect the selected period.

Interpretation
--------------

Stay on ``PinchProblem`` when the study has one active case. Move to
``PinchWorkspace`` when the study itself needs named baseline and variant
cases:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="crude_preheat_train.json",
       project_name="crude_preheat_train",
   )
   workspace.scenario("wide_dt", dt_cont_multiplier=0.5)
   comparison = workspace.compare_cases("baseline", "wide_dt")

Use ``pinch_analysis_service`` only when another application needs a typed
``TargetInput`` to ``TargetOutput`` boundary instead of a live wrapper object.

Next Steps
----------

- :doc:`input-formats-and-validation` for accepted source shapes.
- :doc:`graphing-and-interpretation` for reading curves after a solve.
- :doc:`notebooks-and-sample-cases` for maintained example assets.
- :doc:`../api/pinchproblem` for the full object contract.
