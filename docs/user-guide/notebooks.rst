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

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

Notebook Series
---------------

The packaged notebook series currently includes:

1. ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
2. ``02_total_site_targets_and_sugcc.ipynb``
3. ``03_carnot_hpr_comparison.ipynb``

Each notebook is intentionally comprehensive:

- one real decision question per notebook
- one named plant-style example per workflow
- direct use of the supported public API
- runnable end-to-end as part of the test suite

Recommended Usage
-----------------

For new users, the best learning path is:

1. ``01_basic_pinch_and_dtcont_sensitivity.ipynb`` to learn the baseline pinch workflow, graph reading, and minimum approach sensitivity
2. ``02_total_site_targets_and_sugcc.ipynb`` to compare direct, total-process, and total-site views and inspect the SUGCC
3. ``03_carnot_hpr_comparison.ipynb`` to compare direct and indirect Carnot HPR options over multiple target loads

Interpretation Focus
--------------------

The notebook series is not only about running the solver. It is also intended
to show users how to read the outputs:

- summary metrics such as hot utility, cold utility, heat recovery, and pinch temperatures
- graph outputs such as composite curves, shifted curves, grand composite curves, total-site profiles, and the site utility grand composite curve
- workflow-specific decisions such as whether a ``dt_cont`` assumption is too aggressive, whether total-site targeting changes the answer, and whether a heat-pump or refrigeration target is displacing the right utilities

For a consolidated written guide, see :doc:`interpreting-results`.
For the dedicated helper-backed workflow, see :doc:`heat-pump-targeting`.
For the packaged direct-versus-indirect HPR workflow, see
``03_carnot_hpr_comparison.ipynb`` based on the Chocolate Factory sample.

Design Intent
-------------

The notebook series is treated as part of the tested product surface, not just
documentation. The same sample cases, API methods, and graphing paths used in
the notebooks are covered by automated tests.
