Capability Matrix
=================

This matrix maps the main user capabilities to the preferred public surface,
typical outputs, dependency notes, and support status.

.. list-table::
   :header-rows: 1
   :widths: 22 28 28 12 10

   * - Capability
     - Preferred surface
     - Typical outputs
     - Dependencies
     - Status
   * - Direct heat integration
     - ``PinchProblem.target()``, ``problem.target.direct_heat_integration(...)``
     - utility targets, pinch temperatures, recovery metrics, graphs
     - base
     - Stable
   * - Indirect / Total Site targeting
     - ``problem.target.indirect_heat_integration(...)``
     - Total Process / Total Site utility targets and graph families
     - base
     - Stable
   * - Graphing and HTML export
     - ``problem.plot.*``, ``problem.plot.export(...)``
     - Plotly figures, graph catalog, HTML graph files
     - notebook or dashboard
     - Stable
   * - File-backed workflows
     - ``PinchProblem`` and ``PinchWorkspace``
     - solved results, summary frames, exports, bundle files
     - base; notebook for Excel
     - Stable
   * - Schema-driven execution
     - ``pinch_analysis_service(...)`` with ``TargetInput``
     - typed ``TargetOutput``
     - base
     - Stable
   * - Variable heat-capacity streams
     - ``StreamSchema.segments`` or ``StreamSchema.profile``
     - parent-level targets and reports; explicit ordered segment diagnostics
     - base; synthesis extra for solver-backed HEN design
     - Advanced
   * - Packaged learning assets
     - ``OpenPinch.resources`` and ``openpinch notebook``
     - sample JSON cases and clean notebook sources
     - base; notebook to run notebooks
     - Stable
   * - Heat Pump and refrigeration
     - ``problem.target.direct_heat_pump(...)`` and companions
     - HPR target fields, graph effects, cost fields for simulated cycles
     - notebook/dashboard for graphs
     - Advanced
   * - Direct gas/vapour MVR
     - ``problem.add_component.process_mvr(...)``
     - replacement streams, MVR stage results, changed target summaries
     - notebook/dashboard for graphs
     - Advanced
   * - Exergy post-processing
     - ``problem.target.exergy(...)``
     - exergy metrics and exergetic graph families
     - notebook/dashboard for graphs
     - Advanced
   * - Cogeneration
     - ``problem.target.cogeneration(...)``
     - turbine work, efficiency targets, stage context
     - base
     - Advanced
   * - Heat exchanger network synthesis
     - ``problem.design.enhanced_synthesis_method(quality_tier=...)``
     - ranked networks, selected network, manifest, grid diagrams
     - synthesis extra and solvers
     - Advanced
   * - Community / region framing
     - zone vocabulary and hierarchy support
     - multiscale labels and aggregation structures
     - base
     - Experimental / partial
   * - Lower-level analysis side packages
     - service modules below curated wrappers
     - specialist helper outputs
     - varies
     - Experimental / partial

Support Level Meaning
---------------------

Stable
   Preferred user-facing surfaces documented in guides and curated API pages.

Advanced
   Supported, but intended for users who understand the base thermal target and
   workflow assumptions.

Experimental / partial
   Present in the codebase, but not documented or polished as a primary user
   workflow.

Next Steps
----------

- :doc:`workflow-map` to choose an entrypoint.
- :doc:`support-and-stability` for the detailed support policy.
