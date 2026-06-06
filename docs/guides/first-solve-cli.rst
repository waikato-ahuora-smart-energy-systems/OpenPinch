Working with the CLI
====================

The supported OpenPinch CLI is intentionally narrow. It copies the packaged
notebook series so you can start from maintained learning assets, while the
actual solving, validation, and export workflows live in Python.

Question This Guide Answers
---------------------------

What is the supported OpenPinch CLI surface, and when should I move into
Python?

Supported CLI Workflow
----------------------

Copy the full notebook series:

.. code-block:: bash

   openpinch notebook -o notebooks

Or copy one notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

The current packaged notebooks are:

- ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
- ``02_total_site_targets_and_sugcc.ipynb``
- ``03_carnot_hpr_comparison.ipynb``
- ``04_multistate_targeting_and_state_comparison.ipynb``
- ``05_schema_service_and_output_workflows.ipynb``

When To Leave the CLI
---------------------

Move to the Python workflow immediately when you need to:

- solve a case
- validate the input data
- export Excel or graph artifacts
- work with ``problem.target.*`` or ``problem.plot.*``
- use packaged sample cases directly by name

Next Steps
----------

- For the Python-first workflow, see :doc:`first-solve-python`.
- For file format guidance, see :doc:`input-formats-and-validation`.
- For notebooks and bundled cases, see :doc:`notebooks-and-sample-cases`.
- For graph interpretation, see :doc:`graphing-and-interpretation`.
