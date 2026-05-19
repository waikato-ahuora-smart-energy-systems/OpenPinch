:orphan:

Notebook Workflows
==================

This page has moved, but the packaged notebook workflow still depends on
``python -m pip install "openpinch[notebook]"`` and the current series is:

- ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
- ``02_total_site_targets_and_sugcc.ipynb``
- ``03_carnot_hpr_comparison.ipynb``

These notebooks now load bundled cases directly with
``PinchWorkspace(source="sample_case.json", ...)`` and then use real
``PinchProblem`` cases for targeting, plotting, and comparison.

Use these pages instead:

- :doc:`../guides/notebooks-and-sample-cases`
- :doc:`../examples/notebook-series`
- :doc:`../examples/sample-cases`
