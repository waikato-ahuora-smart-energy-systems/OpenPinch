Graphing and Interpretation
===========================

.. warning::

   Graph owners are currently unsupported internal APIs. Only
   :func:`OpenPinch.main.pinch_analysis_service` is compatibility protected.

Purpose
-------

Use this guide after a case has been solved and you need to connect graph
shape to utility targets, target scope, and workflow decisions.

Prerequisites
-------------

Install ``openpinch[notebook]`` for Plotly figures or
``openpinch[dashboard]`` for the Streamlit review surface.

Sample Case
-----------

Use ``basic_pinch.json`` for process-level graphs and ``pulp_mill.json`` or
``zonal_site.json`` for Total Site profiles and SUGCC views.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json")
   problem.target()

   summary = problem.summary_frame()
   gcc = problem.plot.grand_composite_curve()
   cc = problem.plot.composite_curve()
   catalog = problem.plot.catalog()

Expected Output
---------------

``summary_frame()`` gives the numerical context. ``problem.plot.*`` returns
Plotly figures or graph data for the solved target family. ``catalog()`` helps
confirm which graph families are available before exporting or displaying.

Interpretation
--------------

Use this order:

1. read the summary row and target scope
2. inspect the Grand Composite Curve for utility placement
3. inspect Composite Curves or shifted curves for overlap and pinch behavior
4. inspect Total Site profiles only after confirming the workflow is multizone
5. inspect exergetic graphs only after running exergy post-processing

After exergy enrichment:

.. code-block:: python

   problem.target.exergy()
   gcc_x = problem.plot.exergetic_grand_composite_curve()
   nlp_x = problem.plot.exergetic_net_load_profiles()

For portable review artifacts:

.. code-block:: python

   paths = problem.plot.export("graphs", graph_type="gcc")

Common mistakes are comparing a process-level row to a site-level graph,
reading graph shape before checking utility targets, or treating a graph
change as sufficient without confirming the metrics.

Next Steps
----------

- :doc:`../fundamentals/graphs-and-interpretation` for graph meaning.
- :doc:`exporting-results` for Excel, HTML, and dashboard outputs.
- :doc:`heat-pump-workflows` for HPR graph families.
