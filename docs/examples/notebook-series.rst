Notebook Series
===============

OpenPinch ships with a packaged notebook series that is treated as part of the
supported learning and regression surface. Each notebook is built around one
decision question rather than a generic feature tour. The series now spans the
main single-case API, named multi-case studies, multistate targeting, the
typed/service boundary, the energy-transfer analysis view, and the
vapour-compression plus MVR cascade HPR backend. It also covers direct
gas/vapour stream MVR as a process-component workflow on live workspace cases
and the heat exchanger network design service on a compact four-stream
synthesis case.

Included Notebooks
------------------

``01_basic_pinch_and_dtcont_sensitivity.ipynb``
   Single-case ``PinchProblem`` front door, direct graph interpretation,
   ``area_cost()`` appendix, and ``PinchWorkspace``-based ``dt_cont``
   sensitivity on the real crude preheat train case.

``02_total_site_targets_and_sugcc.ipynb``
   Packaged ``pulp_mill.json`` Total Site workflow, including scope-ladder
   comparison, a real ``Bleaching`` local GCC screen, serialized graph output
   inspection, and local-versus-site cogeneration screens.

``03_carnot_hpr_comparison.ipynb``
   Advanced HPR and refrigeration comparison on ``chocolate_factory.json``
   across the direct heat pump, indirect heat pump, direct refrigeration, and
   indirect refrigeration workflows. The notebook uses the current
   ``HPRcycle.CascadeCarnot`` enum value for broad screening and calls out
   ``HPRcycle.ParallelCarnot`` as the staged Carnot alternative.

``04_multistate_targeting_and_state_comparison.ipynb``
   Real named-state targeting on ``crude_preheat_train_multistate.json`` and
   ``zonal_site_multistate.json``, including ``state_ids``,
   ``target_all_states()``, and cross-state utility tables.

``05_schema_service_and_output_workflows.ipynb``
   ``copy_sample_case(...)``, local-file ``PinchProblem``, typed
   ``TargetInput`` plus ``pinch_analysis_service(...)``, artifact export, and
   serialized ``PinchWorkspace`` variant views.

``06_energy_transfer_analysis.ipynb``
   Energy-transfer targeting on ``pulp_mill.json``, including the
   heat-surplus/deficit table, graph-ready energy-transfer diagram payload,
   standard plot accessor, and Total Site versus local Direct Integration base
   target selection.

``07_vapour_compression_mvr_cascade_hpr.ipynb``
   Heat-pump-only targeting with
   ``HPR_TYPE = "Vapour compression with MVR cascade"``, including the
   configuration fields for VC and MVR stages, split-fraction source/process
   routing, a solved backend result, and the external stream accounting used
   for the combined cascade.

``08_direct_gas_stream_mvr.ipynb``
   Direct gas/vapour process MVR on an in-memory ``PinchWorkspace`` study,
   including baseline, dry MVR, and liquid-injection MVR cases. The notebook
   adds memory-only ``problem.add_component.process_mvr(...)`` components,
   solves direct and Total Site targets after the stream mutation, compares
   target summaries, inspects replacement streams, and toggles the component
   active state.

``09_hen_design_service_four_stream.ipynb``
   Heat exchanger network design-service execution on a compact four-stream
   synthesis problem, including
   ``problem.design.enhanced_synthesis_method(quality_tier=...)`` for tiered
   OpenHENS search, ``problem.design.open_hens_method()`` for the original
   tier-1 route, the ``problem.design.heat_exchanger_network_synthesis(...)``
   dispatcher, workspace workflow dispatch, concise task metadata, and grid
   views for the top networks.

How To Use Them
---------------

Copy the full series:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

Recommended Learning Order
--------------------------

1. Start with the basic pinch and ``dt_cont`` notebook to understand the
   single-case workflow and the named-study sensitivity pattern.
2. Move to the Total Site notebook once you are comfortable with zonal and
   indirect integration concepts.
3. Use the Carnot HPR notebook after you understand how to read the base
   utility and graph outputs.
4. Move to the multistate notebook when your case has named operating states or
   seasonal variation.
5. Use the schema/service notebook when you need typed validation, serialized
   workspace views, or repeatable export workflows.
6. Use the energy-transfer notebook when you need interval-level surplus/deficit
   accounting or a diagram payload derived from an existing thermal target.
7. Use the VC+MVR cascade notebook when you need a refrigerant low stage feeding
   a serial mechanical-vapour-recompression high stage.
8. Use the direct gas/vapour MVR notebook when a process vapour stream itself
   is the recompression source and you need before/after workspace comparison.
9. Use the heat exchanger network design-service notebook when you need network
   synthesis on a compact four-stream case and top-network grid inspection.

Why These Matter
----------------

The notebooks do more than demonstrate commands. They reveal the practical
power of the package: direct single-case solves, named-case comparison,
hierarchical targeting, graph-based interpretation, real multistate studies,
advanced HPR cycle targeting, process-component MVR mutation, heat exchanger
network synthesis, and stable programmatic boundaries built on the same packaged
assets. The distributed copies are also kept output-free so they do not ship
stale plots, tracebacks, or machine-specific execution state.
