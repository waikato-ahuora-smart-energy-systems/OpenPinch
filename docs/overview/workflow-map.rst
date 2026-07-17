Workflow Map
============

One Case
--------

Use :class:`OpenPinch.PinchProblem` when one prepared input should move through
validation, named analysis, cached interpretation, and optional publication.

``source -> PinchProblem -> target/design method -> cached result -> report/plot/export``

Named Cases
-----------

Use :class:`OpenPinch.PinchWorkspace` for a baseline, named unsolved scenarios,
ordered case batches, comparison, and bundle persistence.

``source -> workspace.case/scenario -> case target/design -> compare_cases``

Common Study Paths
------------------

Core heat integration
   ``problem.target.all_heat_integration() -> summary_frame -> core curves``

Focused Total Site
   ``direct_heat_integration -> total_site_heat_integration -> site profiles``

Multiperiod
   ``problem.target.all_periods.<method> -> period_results -> weighted summary``

Heat Pump or refrigeration
   ``named HPR method -> summary -> HPR plots``

Process MVR
   ``components.add_process_mvr -> direct_heat_integration -> compare case``

HEN design
   ``problem.design.<named method> -> design.top -> design.network -> grid``

Observation operations never choose a path. An unsolved scenario remains
unsolved until its target or design method is called.

See :doc:`../examples/tutorial-coverage-map` for every supported operation.
