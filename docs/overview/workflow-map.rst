Workflow Map
============

OpenPinch has several valid entrypoints. Choose by what owns the study state
and how much control you need.

Choose an Entrypoint
--------------------

Use ``PinchProblem`` when:

- you are solving one active case
- inputs come from JSON, Excel, CSV, sample cases, ``TargetInput``, or mappings
- you want validation, targeting, summaries, graphs, exports, period selection,
  advanced target accessors, and HEN design from one object

Use ``PinchWorkspace`` when:

- the study has named baseline and variant cases
- you need case copying, scenario edits, comparisons, or bundle save/load
- an application needs serializable variant views as well as live cases

Use ``pinch_analysis_service`` when:

- another system wants a typed ``TargetInput`` to ``TargetOutput`` boundary
- you do not need wrapper state, period reruns, graph accessors, or exports

Use ``OpenPinch.resources`` when:

- you need to list, inspect, read, or copy packaged sample cases and notebooks
- you want examples without hard-coding repository paths

Use ``openpinch notebook`` when:

- you want clean packaged notebooks copied from a shell
- you do not need the command itself to solve or export anything

Use lower-level services when:

- you need direct access to the prepared ``Zone`` hierarchy
- you are extending OpenPinch or running research-oriented intermediate stages

Workflow Layering
-----------------

.. code-block:: text

   notebooks / scripts / external applications
                 |
                 +--> openpinch notebook
                 +--> OpenPinch.resources
                 |
                 v
   PinchWorkspace / PinchProblem / pinch_analysis_service
                 |
                 v
        validation and data preparation
                 |
                 v
              Zone tree
                 |
                 v
   direct / indirect / HPR / exergy / cogeneration / HEN services
                 |
                 v
       summaries + graph data + exports + design results

Common User Paths
-----------------

First-time solve
   ``PinchProblem("basic_pinch.json") -> validation_report -> target -> summary_frame``

Named sensitivity study
   ``PinchWorkspace(source=...) -> scenario -> compare_cases``

Multiperiod study
   ``problem.period_ids -> problem.target.*(period_id="peak") -> target_all_periods``

Heat Pump screening
   ``base direct/indirect target -> problem.target.direct_heat_pump(...) -> compare summary and graphs``

Heat exchanger network synthesis
   ``PinchProblem("Four-stream-Yee-and-Grossmann-1990-1.json") -> problem.design.enhanced_synthesis_method(quality_tier=2)``

Direct process MVR
   ``workspace.copy_case -> problem.add_component.process_mvr(...) -> re-solve -> compare_cases``

Typed application integration
   ``TargetInput -> pinch_analysis_service(...) -> TargetOutput``

Next Steps
----------

- :doc:`../getting-started` for the first runnable solve.
- :doc:`../guides/index` for task workflows.
- :doc:`../api/index` for exact public contracts.
