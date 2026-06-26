Getting Started
===============

This is the fastest supported path from a clean environment to a solved
OpenPinch case. It uses the packaged ``basic_pinch.json`` sample case so you
can verify the installation before preparing your own data.

Install
-------

Install the base package for core Python workflows:

.. code-block:: bash

   python -m pip install openpinch

Install optional extras only when the workflow needs them:

.. list-table::
   :header-rows: 1

   * - Extra
     - Use when you need
     - Command
   * - base
     - validation, targeting, summaries, schema-driven runs
     - ``python -m pip install openpinch``
   * - notebook
     - Jupyter notebooks, Plotly graph rendering, Excel import/export
     - ``python -m pip install "openpinch[notebook]"``
   * - dashboard
     - Streamlit dashboard review plus graph/export dependencies
     - ``python -m pip install "openpinch[dashboard]"``
   * - synthesis
     - solver-backed heat exchanger network synthesis
     - ``python -m pip install "openpinch[synthesis]"`` then ``idaes get-extensions``
   * - brayton_cycle
     - TESPy-backed Brayton-cycle tooling
     - ``python -m pip install "openpinch[brayton_cycle]"``

OpenPinch currently targets Python 3.14.

Run the First Solve
-------------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="basic_pinch")
   validation = problem.validation_report()
   result = problem.target()
   summary = problem.summary_frame()

   print(validation.valid)
   print(summary[["Target", "Hot Utility Target", "Cold Utility Target"]])

When no local file named ``basic_pinch.json`` exists, ``PinchProblem`` resolves
the packaged sample case that ships with OpenPinch. The same wrapper also
loads JSON files, Excel workbooks, CSV bundles, ``TargetInput`` models, and
plain mappings.

Read the Result
---------------

For a first pass, read the summary table in this order:

1. ``Hot Utility Target``
2. ``Cold Utility Target``
3. ``Heat Recovery``
4. ``Hot Pinch`` and ``Cold Pinch``

Then inspect graphs if you installed the notebook or dashboard extra:

.. code-block:: python

   gcc = problem.plot.grand_composite_curve()
   catalog = problem.plot.catalog()

The Grand Composite Curve is usually the best first graph for utility
placement and Heat Pump screening questions.

Use Named Studies When You Compare Cases
----------------------------------------

Use ``PinchWorkspace`` when the study has a baseline and variants:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="crude_preheat_train.json",
       project_name="crude_preheat_train",
   )
   workspace.scenario("wide_dt", dt_cont_multiplier=0.5)
   comparison = workspace.compare_cases("baseline", "wide_dt")

Use the CLI Only for Notebook Assets
------------------------------------

The supported CLI copies packaged notebooks. It does not solve cases:

.. code-block:: bash

   openpinch notebook -o notebooks

Use Python for validation, solves, graph export, Excel export, dashboards, and
advanced targeting.

Use these pages instead:

- :doc:`guides/first-solve-python`
- :doc:`guides/first-solve-cli`
- :doc:`guides/notebooks-and-sample-cases`
- :doc:`api/cli-and-resources`
- :doc:`overview/workflow-map`
