Capability Matrix
=================

This matrix maps capabilities to the supported contract or an internal owner,
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
     - ``pinch_analysis_service(...)``
     - utility targets, pinch temperatures, recovery metrics, graphs
     - base
     - Stable
   * - Indirect / Total Site targeting
     - ``problem.target.indirect_heat_integration(...)``
     - Total Process / Total Site utility targets and graph families
     - base
     - Unsupported internal
   * - Graphing and HTML export
     - ``problem.plot.*``, ``problem.plot.export(...)``
     - Plotly figures, graph catalog, HTML graph files
     - notebook or dashboard
     - Unsupported internal
   * - File-backed workflows
     - ``PinchProblem`` and ``PinchWorkspace``
     - solved results, summary frames, exports, bundle files
     - base; notebook for Excel
     - Unsupported internal
   * - Schema-driven execution
     - ``OpenPinch.main.pinch_analysis_service(...)`` with a mapping
     - typed ``TargetOutput``
     - base
     - Stable
   * - Variable heat-capacity streams
     - ``StreamSchema`` or ``UtilitySchema`` nested segments/profile
     - parent-level targets and reports; explicit ordered segment diagnostics
     - base; synthesis extra for solver-backed HEN design
     - Unsupported internal
   * - Packaged learning assets
     - ``OpenPinch.resources`` and ``openpinch notebook``
     - sample JSON cases and clean notebook sources
     - base; notebook to run notebooks
     - Repository tooling
   * - Heat Pump and refrigeration
     - ``problem.target.direct_heat_pump(...)`` and companions
     - HPR target fields, graph effects, cost fields for simulated cycles
     - notebook/dashboard for graphs
     - Unsupported internal
   * - Direct gas/vapour MVR
     - ``problem.add_component.process_mvr(...)``
     - replacement streams, MVR stage results, changed target summaries
     - notebook/dashboard for graphs
     - Unsupported internal
   * - Exergy post-processing
     - ``problem.target.exergy(...)``
     - exergy metrics and exergetic graph families
     - notebook/dashboard for graphs
     - Unsupported internal
   * - Cogeneration
     - ``problem.target.cogeneration(...)``
     - turbine work, efficiency targets, stage context
     - base
     - Unsupported internal
   * - Heat exchanger network synthesis
     - ``problem.design.enhanced_synthesis_method(quality_tier=...)``
     - ranked networks, selected network, manifest, grid diagrams
     - synthesis extra and solvers
     - Unsupported internal
   * - Community / region framing
     - zone vocabulary and hierarchy support
     - multiscale labels and aggregation structures
     - base
     - Experimental / partial
   * - Lower-level analysis owner packages
     - concrete analysis and application modules
     - specialist helper outputs
     - varies
     - Experimental / partial

Support Level Meaning
---------------------

Stable
   Compatibility-protected through :mod:`OpenPinch.main`.

Unsupported internal
   Tested repository functionality whose import path and signature may change
   without a compatibility facade.

Repository tooling
   Maintained project tooling that is not a protected Python API.

Experimental / partial
   Present in the codebase, but not documented or polished as a primary user
   workflow.

Next Steps
----------

- :doc:`workflow-map` to choose an entrypoint.
- :doc:`support-and-stability` for the detailed support policy.
