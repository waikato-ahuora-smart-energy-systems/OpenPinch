Notebooks and Sample Cases
==========================

The packaged notebooks and sample cases are part of the supported OpenPinch
learning surface. They are the fastest way to move from a blank environment to
an end-to-end workflow.

Install the notebook runtime first:

.. code-block:: bash

   python -m pip install "openpinch[notebook]"

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

The current packaged notebooks load bundled sample cases directly with
``PinchWorkspace(source="sample_case.json", ...)`` and then work against real
``PinchProblem`` cases inside the workspace. They are packaged as clean sources:
no stored Plotly payloads, no cached execution counts, and no stale traceback
output.

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

Notebook 03 also shows the post-target HPR graph surfaces directly through
``problem.plot.net_load_profiles(zone_name="Direct Heat Pump")`` and
``problem.plot.grand_composite_curve_with_heat_pump(...)``.

Recommended Learning Path
-------------------------

1. `basic_pinch.json` and notebook 01 for baseline workflow and `dt_cont`
   interpretation
2. `zonal_site.json` or `pulp_mill.json` and notebook 02 for Total Site and
   SUGCC workflows
3. `chocolate_factory.json` and notebook 03 for direct-versus-indirect HPR and
   refrigeration comparison
4. `heat_pump_targeting.json` for a smaller direct HPR screening payload when
   you want to test the advanced `problem.target.*` surface without the full
   notebook comparison flow

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
