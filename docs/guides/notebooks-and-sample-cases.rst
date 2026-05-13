Notebooks and Sample Cases
==========================

The packaged notebooks and sample cases are part of the supported OpenPinch
learning surface. They are the fastest way to move from a blank environment to
an end-to-end workflow.

Packaged Sample Cases
---------------------

OpenPinch currently ships with sample cases such as:

- `basic_pinch.json`
- `heat_pump_targeting.json`
- `zonal_site.json`
- `pulp_mill.json`
- `crude_preheat_train.json`
- `chocolate_factory.json`

Copy one into your working directory with:

.. code-block:: bash

   openpinch sample --name basic_pinch.json -o basic_pinch.json

Packaged Notebook Series
------------------------

Copy the full series with:

.. code-block:: bash

   openpinch notebook -o notebooks

Or copy one notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

Current packaged notebooks:

1. `01_basic_pinch_and_dtcont_sensitivity.ipynb`
2. `02_total_site_targets_and_sugcc.ipynb`
3. `03_carnot_hpr_comparison.ipynb`

Recommended Learning Path
-------------------------

1. `basic_pinch.json` and notebook 01 for baseline workflow and `dt_cont`
   interpretation
2. `zonal_site.json` or `pulp_mill.json` and notebook 02 for total-site and
   SUGCC workflows
3. `heat_pump_targeting.json` and notebook 03 for HPR comparison and utility
   displacement logic

Why These Assets Matter
-----------------------

These assets are useful because they:

- exercise the supported public API directly
- provide named examples that align with the docs
- give users a realistic plant-style context instead of toy inputs

Next Steps
----------

- For notebook details, see :doc:`../examples/notebook-series`.
- For sample-case details, see :doc:`../examples/sample-cases`.
