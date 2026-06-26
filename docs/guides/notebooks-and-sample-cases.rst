Notebooks and Sample Cases
==========================

The packaged notebooks and sample cases are part of the supported OpenPinch
learning surface. They are the fastest way to move from a blank environment to
an end-to-end workflow, and the primary discovery surface is Python:

- Python resource helpers expose names, metadata, readers, and copy helpers
- wrapper objects resolve packaged sample-case names directly
- a shell copy shortcut remains documented later for notebook assets

Install the notebook runtime first:

.. code-block:: bash

   python -m pip install "openpinch[notebook]"

Packaged Sample Cases
---------------------

OpenPinch currently ships with sample cases such as:

- ``basic_pinch.json``
- ``crude_preheat_train.json``
- ``crude_preheat_train_multiperiod.json``
- ``zonal_site.json``
- ``zonal_site_multiperiod.json``
- ``pulp_mill.json``
- ``heat_pump_targeting.json``
- ``chocolate_factory.json``
- ``Four-stream-Yee-and-Grossmann-1990-1.json``

Use the resource helpers when you want to inspect or copy them explicitly:

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

You can also load a packaged sample case directly through
``PinchProblem("basic_pinch.json")`` or
``PinchWorkspace(source="basic_pinch.json")`` when no local file with that
name exists. That rule is intentional so local files always win.

Packaged Notebook Series
------------------------

The current packaged notebooks stay on the stable public surfaces while using
real packaged cases or real derivatives of those cases. Notebook 01 starts from
``PinchProblem``, notebooks 01 to 03 use ``PinchWorkspace`` where named study
cases matter, notebook 04 covers named-period targeting, and notebook 05 covers
the typed and serialized boundaries. Notebook 06 covers energy-transfer
analysis outputs. Notebook 07 covers the vapour-compression plus MVR cascade
HPR backend and its split-fraction source/process routing. Notebook 08 covers
the direct gas/vapour MVR process-component workflow, where live
``PinchProblem`` cases are mutated with replacement hot streams and compared
in a ``PinchWorkspace``. Notebook 09 covers the problem-owned heat exchanger
network design service on a compact four-stream synthesis case, including grid
configuration, live solver execution, and top-network inspection. The distributed assets are
packaged as clean sources: no stored Plotly payloads, no cached execution
counts, and no stale traceback output.

Access notebook assets directly from Python:

.. code-block:: python

   from OpenPinch.resources import copy_notebook, list_notebooks, notebook_metadata

   print(list_notebooks())
   print(notebook_metadata("02_total_site_targets_and_sugcc.ipynb").description)
   copy_notebook("01_basic_pinch_and_dtcont_sensitivity.ipynb", "notebooks")

The notebook-copy CLI remains available when you only need to copy source
assets from a shell:

.. code-block:: bash

   openpinch notebook -o notebooks

Current packaged notebooks:

1. ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
2. ``02_total_site_targets_and_sugcc.ipynb``
3. ``03_carnot_hpr_comparison.ipynb``
4. ``04_multiperiod_targeting_and_period_comparison.ipynb``
5. ``05_schema_service_and_output_workflows.ipynb``
6. ``06_energy_transfer_analysis.ipynb``
7. ``07_vapour_compression_mvr_cascade_hpr.ipynb``
8. ``08_direct_gas_stream_mvr.ipynb``
9. ``09_hen_design_service_four_stream.ipynb``

Notebook 04 shows the named-period workflow directly through
``problem.target.direct_heat_integration(period_id="peak")`` and
``problem.target.indirect_heat_integration(period_id="winter")``. Notebook 05
shows the typed ``TargetInput`` boundary and the serialized
``PinchWorkspace`` view layer. Notebook 06 shows
``target.energy_transfer(...)`` with the heat-surplus/deficit table and
``plot.energy_transfer_diagram(...)``. Notebook 07 shows
``target.direct_heat_pump(...)`` with
``HPR_TYPE = "Vapour compression with MVR cascade"`` and the VC+MVR
configuration fields. Notebook 03 uses
``HPR_TYPE = "Cascade Carnot cycles"`` for broad direct/indirect
screening and notes ``"Parallel Carnot cycles"`` as the explicit staged
Carnot option. Notebook 08 shows ``add_component.process_mvr(...)``,
``stage_results_by_period``, replacement stream inspection, component
activation/deactivation, and baseline-versus-MVR comparison through
``workspace.compare_cases(...)``. Notebook 09 shows
``problem.design.enhanced_synthesis_method(quality_tier=...)``,
``problem.design.open_hens_method()`` for original tier 1,
``problem.design.heat_exchanger_network_synthesis(method=...)``, and
``workspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")``
with a small heat exchanger network design grid and grid views for the top
networks.

Recommended Learning Path
-------------------------

1. ``basic_pinch.json`` and notebook 01 for the single-case workflow and
   ``dt_cont`` interpretation.
2. ``zonal_site.json`` or ``pulp_mill.json`` and notebook 02 for Total Site
   and SUGCC workflows.
3. ``chocolate_factory.json`` and notebook 03 for
   direct-versus-indirect HPR and refrigeration comparison.
4. ``crude_preheat_train_multiperiod.json`` and
   ``zonal_site_multiperiod.json`` with notebook 04 for real named-period
   comparison.
5. ``basic_pinch.json`` and notebook 05 when you need typed validation,
   exports, or serialized workspace views.
6. ``pulp_mill.json`` and notebook 06 when you need energy-transfer diagrams
   or interval surplus/deficit accounting.
7. ``heat_pump_targeting.json`` and notebook 07 when you need the
   vapour-compression plus MVR cascade HPR backend.
8. Notebook 08 when you need direct gas/vapour MVR on selected process streams
   before re-solving direct and Total Site targets.
9. Notebook 09 when you need heat exchanger network design-service execution
   and network inspection on the converted OpenHENS Ye and Grossman four-stream
   synthesis case.

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
