Notebook Workflows
==================

OpenPinch ships with a packaged notebook series so users can start from a
working workflow rather than a blank notebook.

Copy The Notebook Series
------------------------

Use the CLI to copy the notebooks into your working directory:

.. code-block:: bash

   openpinch notebook -o notebooks

You can also copy a single notebook:

.. code-block:: bash

   openpinch notebook --name 04_heat_pump_workflow.ipynb -o notebooks

Notebook Series
---------------

The packaged notebook series currently includes:

1. ``01_basic_pinch_analysis.ipynb``
2. ``02_graphs_and_interpretation.ipynb``
3. ``03_zonal_analysis.ipynb``
4. ``04_heat_pump_workflow.ipynb``
5. ``05_batch_comparison.ipynb``

Each notebook is intentionally narrow:

- one workflow per notebook
- one known-good example or output pattern
- direct use of the supported public API
- runnable end-to-end as part of the test suite

Recommended Usage
-----------------

For new users, the best learning path is:

1. ``01_basic_pinch_analysis.ipynb`` to learn the solve-and-summarize flow
2. ``02_graphs_and_interpretation.ipynb`` to understand graph outputs
3. ``03_zonal_analysis.ipynb`` if your work involves nested zones or total-site studies
4. ``04_heat_pump_workflow.ipynb`` for heat-pump targeting and integration
5. ``05_batch_comparison.ipynb`` for comparing multiple cases programmatically

Interpretation Focus
--------------------

The notebook series is not only about running the solver. It is also intended
to show users how to read the outputs:

- summary metrics such as hot utility, cold utility, heat recovery, and pinch temperatures
- graph outputs such as composite curves, shifted curves, grand composite curves, and total-site profiles
- workflow-specific decisions such as whether a heat-pump integration scenario is displacing the right utilities

For a consolidated written guide, see :doc:`interpreting-results`.
For the dedicated helper-backed workflow, see :doc:`heat-pump-targeting`.

Design Intent
-------------

The notebook series is treated as part of the tested product surface, not just
documentation. The same sample cases, API methods, and graphing paths used in
the notebooks are covered by automated tests.
