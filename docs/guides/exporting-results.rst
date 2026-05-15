Exporting Results
=================

OpenPinch supports several output surfaces depending on whether you want a
quick check, a report artifact, or an interactive review.

Question This Guide Answers
---------------------------

How do I get OpenPinch results out of the runtime object and into a form I can
review or share?

Main Output Surfaces
--------------------

Terminal summary
   Best for quick comparison and regression-style checking.

`summary_frame()`
   Best for Python-side inspection and downstream data manipulation.

Excel export
   Best for detailed review and handoff.

Graph HTML export
   Best for visual sharing outside Python.

Dashboard
   Best for interactive inspection once a case is already solved.

Python Examples
---------------

.. code-block:: python

   summary = problem.summary_frame()
   detailed = problem.summary_frame(detailed=True)
   workbook = problem.export_excel("results")
   graphs = problem.plot.export("graphs", graph_type="gcc")

CLI Examples
------------

.. code-block:: bash

   openpinch run basic_pinch.json -o results --graph-output graphs
   openpinch graph basic_pinch.json --graph-type composite -o graphs

Choosing the Right Output
-------------------------

Use `summary_frame()` when:

- you want a scriptable table
- you are comparing scenarios in code

Use Excel when:

- you want a reviewable report artifact
- the audience prefers spreadsheet consumption

These workbook-oriented outputs require the ``openpinch[notebook]`` or
``openpinch[dashboard]`` extra.

Use HTML graphs when:

- you want portable visual output
- you do not need the live Python object

These rendered graph exports require the ``openpinch[notebook]`` or
``openpinch[dashboard]`` extra.

Use the dashboard when:

- you want an interactive review after solving

This surface requires ``openpinch[dashboard]``.

Next Steps
----------

- For graph usage, see :doc:`graphing-and-interpretation`.
- For the exact wrapper methods, see :doc:`../api/pinchproblem`.
