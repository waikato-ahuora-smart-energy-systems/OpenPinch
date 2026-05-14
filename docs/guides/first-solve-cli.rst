First Solve with the CLI
========================

Use the CLI when you want the shortest path from a file-backed case to a
validated answer, printed summary, and export artifacts.

Question This Guide Answers
---------------------------

How do I run a supported OpenPinch workflow from the terminal without writing
Python?

Recommended Inputs
------------------

The fastest supported path is a packaged sample case:

.. code-block:: bash

   openpinch sample --name basic_pinch.json -o basic_pinch.json

You can also point the CLI at:

- a JSON problem payload
- a workbook file
- a CSV bundle directory

Step 1. Validate the Case
-------------------------

.. code-block:: bash

   openpinch validate basic_pinch.json

Use this first when you want to separate data issues from analysis issues.

Step 2. Run the Analysis
------------------------

.. code-block:: bash

   openpinch run basic_pinch.json --graph-output graphs -o results

This does three useful things at once:

- prints a summary table to the terminal
- exports an Excel results workbook
- exports graph HTML files

Step 3. Export Specific Graphs
------------------------------

When you only want graphs:

.. code-block:: bash

   openpinch graph basic_pinch.json --graph-type gcc -o graphs

Good first graph choices are:

- `gcc`
- `composite`
- `shifted`

Step 4. Copy Learning Assets
----------------------------

Packaged notebooks:

.. code-block:: bash

   openpinch notebook -o notebooks

Packaged samples:

.. code-block:: bash

   openpinch sample --name pulp_mill.json -o pulp_mill.json

When To Leave the CLI
---------------------

Move to the Python workflow when you need:

- repeated scenario studies in one session
- direct access to summary frames and graph objects
- `problem.target.*` workflows
- `problem.plot.*` workflows

Next Steps
----------

- For the Python-first workflow, see :doc:`first-solve-python`.
- For file format guidance, see :doc:`input-formats-and-validation`.
- For graph interpretation, see :doc:`graphing-and-interpretation`.
