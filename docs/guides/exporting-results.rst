Exporting Results
=================

Purpose
-------

Use this guide when you need to move solved OpenPinch results into tables,
workbooks, HTML graphs, or an interactive dashboard.

Prerequisites
-------------

Run a solve first. Install ``openpinch[notebook]`` for Excel and graph exports
or ``openpinch[dashboard]`` for the Streamlit dashboard.

Sample Case
-----------

Use ``basic_pinch.json`` for first exports. Use ``pulp_mill.json`` when you
want Total Site and cogeneration outputs in the same workbook.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json")
   problem.target.all_heat_integration()

   summary = problem.summary_frame()
   detailed = problem.summary_frame(detailed=True)
   workbook_path = problem.export_excel("results")
   graph_paths = problem.plot.export(
       "graphs",
       plot=problem.plot.grand_composite_curve,
   )

Expected Output
---------------

- ``summary_frame()`` returns a pandas table for scriptable inspection.
- ``export_excel(...)`` writes workbook artifacts for review or handoff.
- ``problem.plot.export(...)`` writes portable HTML graph files.
- ``show_dashboard()`` opens the Streamlit review surface when dashboard
  dependencies are installed.

Interpretation
--------------

Choose the output by audience:

- Use summary frames for Python analysis and regression checks.
- Use Excel when the review audience expects spreadsheet artifacts.
- Use HTML graphs when visual interpretation needs to be shared outside
  Python.
- Use the dashboard when a solved case needs interactive inspection.

The CLI does not provide export commands. Use Python for solved outputs.

Next Steps
----------

- :doc:`graphing-and-interpretation` for graph reading order.
- :doc:`../api/pinchproblem` for the exact export methods.
- :doc:`../api/cli-and-resources` for the boundary between Python and CLI.
