Notebook Series
===============

OpenPinch ships a maintained notebook series for learning and regression. The
notebooks are advanced repository examples: they deliberately exercise
unsupported internal owners and say so in their opening cell. Only the
``pinch_analysis_service(...)`` section in notebook 09 demonstrates the
compatibility-protected Python contract.

Included Notebooks
------------------

``01_first_solve_summary_graphs.ipynb``
   Foundation notebook for a single ``PinchProblem`` solve, validation,
   summary tables, graph interpretation, ``area_cost()`` output, and a
   ``PinchWorkspace`` ``dt_cont`` sensitivity study on a real crude preheat
   train case.

``02_total_site_sugcc_interpretation.ipynb``
   Method notebook for ``pulp_mill.json`` Total Site analysis, local versus
   site targeting, a real ``Bleaching`` local GCC screen, serialized graph data
   inspection, SUGCC interpretation, and cogeneration context.

``03_multiperiod_workspace_scenarios.ipynb``
   Scenario notebook for ``crude_preheat_train_multiperiod.json`` and
   ``zonal_site_multiperiod.json``, including ``period_ids``,
   ``target_all_periods()``, period-specific direct and indirect integration,
   and cross-period utility comparison.

``04_carnot_heat_pump_screening.ipynb``
   Advanced-method notebook for direct and indirect Heat Pump screening on
   ``chocolate_factory.json``. It uses the internal ``problem.target.*`` and
   ``problem.plot.*`` surfaces with ``HPRcycle.CascadeCarnot`` and
   ``HPRcycle.ParallelCarnot`` options.

``05_direct_gas_stream_mvr_scenarios.ipynb``
   Advanced-method notebook for direct gas/vapour process MVR on an in-memory
   ``PinchWorkspace`` study. It compares baseline, dry MVR, and
   liquid-injection MVR cases, inspects replacement streams, and toggles the
   process component active state.

``06_vapour_compression_mvr_cascade_hpr.ipynb``
   Method notebook for the VC+MVR cascade HPR backend, including the
   configuration fields for VC and MVR stages, standalone MVR thermodynamics,
   stream profiles, graph interpretation, and external stream accounting.

``07_heat_exchanger_network_synthesis.ipynb``
   Advanced-method notebook for the internal HEN design accessors on the compact
   four-stream Yee and Grossmann benchmark. It covers
   ``problem.design.enhanced_synthesis_method(...)``,
   ``problem.design.open_hens_method()``, ranked network selection, manifests,
   and grid diagrams.

``08_energy_transfer_analysis.ipynb``
   Method notebook for energy-transfer targeting on ``pulp_mill.json``,
   including heat-surplus/deficit tables, graph-ready energy-transfer diagram
   data, standard plot accessors, and Total Site versus local Direct
   Integration target selection.

``09_schema_service_exports_and_bundles.ipynb``
   Integrator notebook for ``copy_sample_case(...)``, local-file
   ``PinchProblem`` loading, typed ``TargetInput`` requests,
   ``pinch_analysis_service(...)``, Excel/graph export, workspace variant
   views, and bundle persistence.

``10_multiperiod_hpr_shared_design.ipynb``
   Focused notebook for opt-in weighted multiperiod HPR design on
   ``crude_preheat_train_multiperiod.json``. It contrasts period-specific HPR
   optima with one shared Cascade Carnot design, inspects ``hpr_details``, and
   uses weighted summary modes.

How To Use Them
---------------

Copy the full series:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_sugcc_interpretation.ipynb -o notebooks

Recommended Paths
-----------------

I want to solve a case with advanced methods
   Work through notebooks 01, 03, 04, 05, 07, and 10. This path starts with
   the single-case solve and summary workflow, then moves into named periods,
   Heat Pump screening, process MVR case mutation, HEN synthesis, and shared
   multiperiod HPR design.

I need to understand the method
   Work through notebooks 02, 06, and 08. This path explains Total Site and
   SUGCC interpretation, VC+MVR cascade mechanics, and interval-level
   energy-transfer accounting.

I am integrating or extending OpenPinch
   Start with notebook 09 and pair it with :doc:`../api/index`. Add notebook
   07 when your integration touches HEN synthesis, ranked network inspection,
   or solver-backed design workflows.

Why These Matter
----------------

The notebooks do more than demonstrate commands. They reveal the practical
power of the package: direct single-case solves, named-case comparison,
hierarchical targeting, graph-based interpretation, real multiperiod studies,
advanced HPR cycle targeting, weighted shared HPR design, process-component
MVR mutation, heat exchanger network synthesis, and the protected main-service
boundary built on the same packaged assets. The distributed copies are kept
output-free so they do not ship stale plots, tracebacks, or machine-specific
execution state.
