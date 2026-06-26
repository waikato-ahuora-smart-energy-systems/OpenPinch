Notebooks and Sample Cases
==========================

Purpose
-------

Use packaged notebooks and sample cases when you want maintained, reproducible
learning assets that exercise the public OpenPinch API.

Prerequisites
-------------

Install the notebook extra before running copied notebooks:

.. code-block:: bash

   python -m pip install "openpinch[notebook]"

Sample Cases
------------

OpenPinch currently ships:

- ``Four-stream-Yee-and-Grossmann-1990-1.json``
- ``basic_pinch.json``
- ``chocolate_factory.json``
- ``crude_preheat_train.json``
- ``crude_preheat_train_multiperiod.json``
- ``heat_pump_targeting.json``
- ``pulp_mill.json``
- ``zonal_site.json``
- ``zonal_site_multiperiod.json``

Runnable Workflow
-----------------

Discover sample cases from Python:

.. code-block:: python

   from OpenPinch.resources import (
       copy_sample_case,
       list_sample_cases,
       read_sample_case,
       sample_case_metadata,
   )

   print(list_sample_cases())
   print(sample_case_metadata("basic_pinch.json").description)
   print(read_sample_case("basic_pinch.json")[:120])
   copy_sample_case("basic_pinch.json", "basic_pinch.json")

Load a packaged sample case directly when no local file with that name exists:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

   problem = PinchProblem("basic_pinch.json")
   workspace = PinchWorkspace(source="crude_preheat_train.json")

Copy notebooks from Python:

.. code-block:: python

   from OpenPinch.resources import copy_notebook, list_notebooks, notebook_metadata

   print(list_notebooks())
   print(notebook_metadata("02_total_site_targets_and_sugcc.ipynb").description)
   copy_notebook("01_basic_pinch_and_dtcont_sensitivity.ipynb", "notebooks")

Copy notebooks from the shell:

.. code-block:: bash

   openpinch notebook -o notebooks

Expected Output
---------------

Packaged notebooks are copied as clean sources: no stored Plotly data, no
cached execution counts, and no stale traceback output.

The current notebook series is:

1. ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
2. ``02_total_site_targets_and_sugcc.ipynb``
3. ``03_carnot_hpr_comparison.ipynb``
4. ``04_multiperiod_targeting_and_period_comparison.ipynb``
5. ``05_schema_service_and_output_workflows.ipynb``
6. ``06_energy_transfer_analysis.ipynb``
7. ``07_vapour_compression_mvr_cascade_hpr.ipynb``
8. ``08_direct_gas_stream_mvr.ipynb``
9. ``09_hen_design_service_four_stream.ipynb``

Interpretation
--------------

Recommended learning path:

1. ``basic_pinch.json`` and notebook 01 for first solves and ``dt_cont`` sensitivity.
2. ``zonal_site.json`` or ``pulp_mill.json`` and notebook 02 for Total Site and SUGCC.
3. ``chocolate_factory.json`` and notebook 03 for direct-versus-indirect HPR.
4. multiperiod sample cases and notebook 04 for named-period targeting.
5. notebook 05 for typed ``TargetInput`` and serialized workspace views.
6. notebook 06 for energy-transfer diagrams and interval surplus/deficit tables.
7. notebook 07 for vapour-compression plus MVR cascade HPR.
8. notebook 08 for direct gas/vapour MVR process components.
9. notebook 09 for heat exchanger network synthesis and top-network inspection.

Next Steps
----------

- :doc:`../examples/notebook-series` for notebook-by-notebook details.
- :doc:`../examples/sample-cases` for sample-case descriptions.
- :doc:`../api/cli-and-resources` for the exact resource helper API.
