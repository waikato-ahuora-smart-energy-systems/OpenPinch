Capability Matrix
=================

This page maps the main OpenPinch capabilities to their primary user-facing
surfaces and outputs.

.. list-table::
   :header-rows: 1
   :widths: 22 28 28 22

   * - Capability
     - Main User Surface
     - Typical Outputs
     - Status
   * - Direct heat integration targeting
     - ``PinchProblem.target()``, ``problem.target.direct_heat_integration(...)``
     - utility targets, pinch temperatures, recovery metrics, graphs
     - Stable
   * - Indirect / Total Site targeting
     - ``problem.target.indirect_heat_integration(...)``,
       multiscale service layer
     - Total Process / Total Site utility targets and graphs
     - Stable
   * - Graph generation
     - ``problem.plot.*``, ``problem.plot.export(...)``
     - composite curves, GCC, site profiles, HTML exports
     - Stable
   * - File-backed workflows
     - JSON, Excel, CSV bundle loading via ``PinchProblem.load(...)``
     - solved `TargetOutput`, summary frames, Excel export
     - Stable
   * - Schema-driven workflows
     - :func:`OpenPinch.main.pinch_analysis_service`,
       :class:`OpenPinch.lib.schemas.io.TargetInput`
     - typed programmatic solve results
     - Stable
   * - Direct and indirect HPR targeting
     - ``problem.target.direct_heat_pump(...)``,
       ``problem.target.indirect_heat_pump(...)``,
       refrigeration companions
     - target models with HPR summary fields and graph effects
     - Advanced
   * - Cogeneration targeting
     - ``problem.target.cogeneration(...)``,
       service-layer cogeneration entrypoint
     - turbine work and efficiency targets
     - Advanced
   * - Exergy targeting post-processing
     - ``problem.target.exergy(...)``,
       exergetic plot accessors
     - exergy metrics plus exergetic GCC and NLP graphs
     - Advanced
   * - Area / cost targeting
     - ``problem.target.area_cost(...)``,
       common area/cost helpers
     - refreshed DI targets with area/cost context
     - Advanced
   * - Packaged learning assets
     - ``openpinch notebook``, :mod:`OpenPinch.resources`
     - runnable sample JSONs and notebooks
     - Stable
   * - Community / region framing
     - zone-type vocabulary and hierarchy support
     - extended multiscale labeling and aggregation structures
     - Experimental / partial
   * - Energy transfer and lower-level analysis side packages
     - lower-level service modules
     - specialist analysis helpers
     - Experimental / partial

How To Read This Matrix
-----------------------

Stable
   Supported as part of the primary package workflow and documented as a public
   user-facing surface.

Advanced
   Supported, but best used after you understand the fundamentals and core
   workflows. These surfaces are more specialized and often require closer
   interpretation.

Experimental / partial
   Exposed in the codebase, but not yet as fully documented or workflow-polished
   as the core surfaces.

Package Strengths To Keep In Mind
---------------------------------

The practical power of OpenPinch comes from the combination of:

- multiple workflow styles over one analysis engine
- hierarchical zone modeling
- graph-first interpretation support
- typed input/result models for integration into other software
- packaged examples and notebooks that exercise the supported public API

For the workflow-level view, continue to :doc:`workflow-map`.
