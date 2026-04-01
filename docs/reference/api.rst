API Reference
=============

OpenPinch exposes a deliberately layered API. Most workflows can stay at the
top layer, while advanced integrations can drop into the underlying zone,
problem-table, and targeting modules when they need more control.

Where To Start
--------------

- Use :class:`~OpenPinch.classes.pinch_problem.PinchProblem` when you want a
  notebook- or script-friendly wrapper that can load JSON, Excel, or CSV input,
  run the analysis, and export results.
- Use :func:`~OpenPinch.main.pinch_analysis_service` when you are integrating
  OpenPinch into another application and want a typed service boundary based on
  :class:`~OpenPinch.lib.schema.TargetInput` and
  :class:`~OpenPinch.lib.schema.TargetOutput`.
- Use :func:`~OpenPinch.analysis.data_preparation.prepare_problem` together
  with :func:`~OpenPinch.main.get_targets` when you need to inspect or mutate
  the intermediate :class:`~OpenPinch.classes.zone.Zone` tree directly.
- Use :mod:`OpenPinch.streamlit_webviewer.web_graphing` when you already have a
  solved zone hierarchy and want to embed OpenPinch plots or a Streamlit-based
  dashboard.

Common Data Contracts
---------------------

The reference pages repeatedly refer to the same small set of core types:

- :class:`~OpenPinch.lib.schema.TargetInput` is the validated request model for
  programmatic analysis.
- :class:`~OpenPinch.classes.zone.Zone` is the in-memory hierarchy built during
  preparation and then populated with streams, utilities, targets, and graphs.
- :class:`~OpenPinch.classes.energy_target.EnergyTarget` stores the solved
  metrics for one direct-integration, total-zone, or total-site target.
- :class:`~OpenPinch.classes.problem_table.ProblemTable` is the numerical table
  behind composite curves, pinch temperatures, and utility cascades.
- :class:`~OpenPinch.lib.schema.TargetOutput` is the structured response
  returned by the high-level service layer.

Reference Map
-------------

- :doc:`api-core` covers the supported entry points most users should start
  from.
- :doc:`api-analysis` documents the preparation and targeting algorithms that
  execute after validation.
- :doc:`api-classes` explains the domain objects created and returned by the
  analysis pipeline.
- :doc:`api-lib` documents configuration, enums, and Pydantic schemas that make
  up the typed wire format.
- :doc:`api-utils` covers import/export helpers, numerical utilities, and
  visualisation support.

.. toctree::
   :maxdepth: 1

   api-core
   api-analysis
   api-classes
   api-lib
   api-utils
