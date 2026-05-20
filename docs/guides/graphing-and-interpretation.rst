Graphing and Interpretation
===========================

This guide focuses on the practical use of OpenPinch graph outputs after a
case has been solved.

Question This Guide Answers
---------------------------

Which graph should I inspect first, and how do I connect graph changes back to
the summary metrics?

Fastest Graph Workflow
----------------------

Python:

.. code-block:: python

   gcc = problem.plot.grand_composite_curve()
   cc = problem.plot.composite_curve()

Best Default Graph
------------------

If you only inspect one graph after the summary, inspect the grand composite
curve.

It is usually the best graph for:

- utility placement questions
- residual thermal pocket interpretation
- Heat Pump opportunity screening

Recommended Interpretation Order
--------------------------------

1. read the summary table first
2. identify the target row and scope
3. inspect the GCC
4. move to composite or shifted composite curves if you need overlap detail
5. move to site-level graph families only when the workflow is multiscale

Exporting Graphs
----------------

Use Python when you want direct `plotly` figures. Install
``openpinch[notebook]`` or ``openpinch[dashboard]`` first.

Use `problem.plot.export(...)` when you want portable HTML output for sharing
or review outside Python.

Common Mistakes
---------------

- reading a graph without checking the target scope first
- treating a graph improvement as enough without checking the utility numbers
- comparing process-level and site-level views as though they were the same
  question

Next Steps
----------

- For graph meaning, see :doc:`../fundamentals/graphs-and-interpretation`.
- For multiscale workflows, see :doc:`zonal-and-total-site-workflows`.
