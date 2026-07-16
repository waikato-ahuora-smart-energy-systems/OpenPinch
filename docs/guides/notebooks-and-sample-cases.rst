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
   print(notebook_metadata("01_first_solve_summary_graphs.ipynb").description)
   copy_notebook("01_first_solve_summary_graphs.ipynb", "notebooks")

Copy notebooks from the shell:

.. code-block:: bash

   openpinch notebook -o notebooks

Expected Output
---------------

Packaged notebooks are copied as clean sources: no stored Plotly data, no
cached execution counts, and no stale traceback output.

The current notebook series is:

1. ``01_first_solve_summary_graphs.ipynb``
2. ``02_total_site_sugcc_interpretation.ipynb``
3. ``03_multiperiod_workspace_scenarios.ipynb``
4. ``04_carnot_heat_pump_screening.ipynb``
5. ``05_direct_gas_stream_mvr_scenarios.ipynb``
6. ``06_vapour_compression_mvr_cascade_hpr.ipynb``
7. ``07_heat_exchanger_network_synthesis.ipynb``
8. ``08_energy_transfer_analysis.ipynb``
9. ``09_schema_service_exports_and_bundles.ipynb``
10. ``10_multiperiod_hpr_shared_design.ipynb``

Interpretation
--------------

Use the series according to the work you are doing, not just by notebook
number.

I want to solve a case with advanced methods
   Start with ``01_first_solve_summary_graphs.ipynb`` for the single-case
   solve, summary, graph, and ``dt_cont`` sensitivity pattern. Move to
   ``03_multiperiod_workspace_scenarios.ipynb`` when operating periods matter.
   Use ``04_carnot_heat_pump_screening.ipynb`` for direct/indirect heat-pump
   screening, ``05_direct_gas_stream_mvr_scenarios.ipynb`` for process MVR
   case mutation, and ``07_heat_exchanger_network_synthesis.ipynb`` for HEN
   synthesis and ranked network inspection. Use
   ``10_multiperiod_hpr_shared_design.ipynb`` when one HPR design must be
   optimised across several weighted periods.

I need to understand the method
   Use ``02_total_site_sugcc_interpretation.ipynb`` to connect local targets,
   Total Site targets, SUGCC profiles, and cogeneration screens. Use
   ``06_vapour_compression_mvr_cascade_hpr.ipynb`` to understand the
   VC+MVR cascade mechanics and ``08_energy_transfer_analysis.ipynb`` for
   interval surplus/deficit accounting and energy-transfer diagrams.

I am integrating or extending OpenPinch
   Use ``09_schema_service_exports_and_bundles.ipynb`` for typed
   ``TargetInput`` requests, ``pinch_analysis_service(...)``, exports,
   workspace variant views, and bundle persistence. Pair it with
   :doc:`../api/index` when you need public contract details, and use
   ``07_heat_exchanger_network_synthesis.ipynb`` when extending synthesis
   workflows.

Next Steps
----------

- :doc:`../examples/notebook-series` for notebook-by-notebook details.
- :doc:`../examples/sample-cases` for sample-case descriptions.
- :doc:`../api/cli-and-resources` for the exact resource helper API.
