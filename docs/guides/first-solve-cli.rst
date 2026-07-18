Working with the CLI
====================

Purpose
-------

Use the OpenPinch CLI when you want to copy maintained notebook sources into a
working directory. The CLI is an onboarding and asset-copy surface only.
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

``01_first_solve_summary_graphs.ipynb``
   First solve, summary tables, graphing, and workspace sensitivity.

Runnable Workflow
-----------------

Copy the full notebook series:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 01_first_solve_summary_graphs.ipynb -o notebooks

The packaged notebook series is:

- ``01_first_solve_summary_graphs.ipynb``
- ``02_total_site_sugcc_interpretation.ipynb``
- ``03_multiperiod_workspace_scenarios.ipynb``
- ``04_carnot_heat_pump_screening.ipynb``
- ``05_direct_gas_stream_mvr_scenarios.ipynb``
- ``06_vapour_compression_mvr_cascade_hpr.ipynb``
- ``07_multiperiod_heat_exchanger_network.ipynb``
- ``08_energy_transfer_analysis.ipynb``
- ``09_schema_service_exports_and_bundles.ipynb``
- ``10_multiperiod_hpr_shared_design.ipynb``

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
