Capability Matrix
=================

.. list-table::
   :header-rows: 1
   :widths: 23 35 22 20

   * - Capability
     - Public surface
     - Typical result
     - Dependency
   * - Direct and Total Site heat integration
     - ``problem.target.direct_heat_integration()``,
       ``total_site_heat_integration()``, ``all_heat_integration()``
     - utility, recovery, Pinch, and site targets
     - base
   * - Multiperiod heat integration
     - ``problem.target.all_periods.*``
     - ordered period outputs and weighted summaries
     - base
   * - Variable heat-capacity streams
     - segmented mapping input through ``PinchProblem``
     - one physical stream with ordered thermal segments
     - base
   * - Heat Pump and refrigeration
     - named Carnot, vapour-compression, Brayton, and MVR target methods
     - HPR targets, costs, and graphs
     - model-specific HPR extras
   * - Process MVR
     - ``problem.components.add_process_mvr()``
     - replacement streams and stage results
     - HPR extras
   * - Area/cost, exergy, and energy transfer
     - named ``problem.target`` enrichment methods
     - enriched target and graph families
     - base; plotting for figures
   * - Cogeneration
     - default and named turbine-model target methods
     - work and efficiency targets
     - base
   * - HEN synthesis
     - named ``problem.design`` methods and design-result view
     - ranked networks, selected network, utilities, and grid
     - HEN solver
   * - Named scenarios and case batches
     - ``workspace.scenario()``, ``workspace.cases(...)``
     - ordered per-case results and comparisons
     - method-specific
   * - Reports, plots, and exports
     - cached observation plus explicit output methods
     - frames, mappings, reports, figures, files, dashboard
     - output-specific
   * - Packaged learning assets
     - ``OpenPinch.resources`` and ``openpinch notebook``
     - sample cases and eighteen clean notebooks
     - base; optional profiles as declared

All workflow classes and accessors in this matrix are supported package
surfaces. Contributor analysis modules remain implementation owners rather than
alternative process-engineer entry points.

See :doc:`../examples/tutorial-coverage-map` for exact operation coverage.
