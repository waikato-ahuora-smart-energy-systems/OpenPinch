Working with the CLI
====================

Purpose
-------

Use the OpenPinch CLI when you want to copy maintained notebook sources into a
working directory, or write the contract JSON Schemas with
``openpinch schema -o DIR``. The CLI is an onboarding and asset-copy surface
only.
Validation, targeting, graph export, Excel export, dashboards, and advanced
workflows are Python workflows.

Prerequisites
-------------

Install the package and, if you plan to run notebooks, the notebook extra:

.. code-block:: bash

   python -m pip install openpinch
   python -m pip install "openpinch[notebook]"

Sample Asset
------------

Notebook 01 is the recommended first copied asset:

``01_first_solve_and_core_curves.ipynb``
   First solve, summary tables, and core curves.

Runnable Workflow
-----------------

Copy the full notebook series:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 01_first_solve_and_core_curves.ipynb -o notebooks/

The packaged notebook series has nineteen notebooks, ``01_first_solve_and_core_curves.ipynb``
through ``19_utility_placement_optimisation.ipynb``; see
:doc:`../examples/notebook-series` for the full list.

Expected Output
---------------

The command writes clean notebook source files into the output directory. The
packaged notebooks ship without execution counts, stored Plotly data, cached
tracebacks, or hidden local-state assumptions.

Interpretation
--------------

Move to Python as soon as you need to:

- solve a case
- validate source data
- export Excel or HTML graph artifacts
- use ``PinchProblem`` or ``PinchWorkspace``
- call ``problem.target.*``, ``problem.plot.*``, or ``problem.design.*``

OpenPinch intentionally does not provide CLI commands for solving, graph
export, or validation.

Next Steps
----------

- :doc:`first-solve-python` for the canonical solve workflow.
- :doc:`notebooks-and-sample-cases` for Python resource helpers.
- :doc:`../api/cli-and-resources` for the exact CLI and asset APIs.
