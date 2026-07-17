Workflow Map
============

OpenPinch has one supported Python entrypoint and several unsupported internal
owners used by repository applications and advanced notebooks.

Choose an Entrypoint
--------------------

Use ``OpenPinch.main.pinch_analysis_service`` when:

- another system wants the compatibility-protected request/response boundary
- you do not need live wrapper state or internal graph/export helpers

Use ``PinchProblem`` internally when:

- you are solving one active case
- inputs come from JSON, Excel, CSV, sample cases, ``TargetInput``, or mappings
- you want validation, targeting, summaries, graphs, exports, period selection,
  advanced target accessors, and HEN design from one object

Use ``PinchWorkspace`` internally when:

- the study has named baseline and variant cases
- you need case copying, scenario edits, comparisons, or bundle save/load
- an application needs serializable variant views as well as live cases

Use ``OpenPinch.resources`` as repository tooling when:

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

   external applications
                 |
                 v
        OpenPinch.main.pinch_analysis_service
                 |
                 v
          unsupported internal owners
   application / analysis / domain / contracts
   adapters / optimisation / presentation
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
   ``mapping -> pinch_analysis_service(...) -> serialized result``

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
   ``mapping -> pinch_analysis_service(...) -> target output``

Next Steps
----------

- :doc:`../getting-started` for the first runnable solve.
- :doc:`../guides/index` for task workflows.
- :doc:`../api/package-root` for the exact external contract.
