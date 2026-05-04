Core API
========

The core API is the supported front door to OpenPinch. It is intentionally
small: one high-level service function, one convenience wrapper class, and a
small number of lower-level orchestration helpers for callers that need to
operate on prepared zone trees directly.

Recommended Usage
-----------------

For new code, prefer one of these two patterns:

1. Build a :class:`~OpenPinch.lib.schema.TargetInput` payload and call
   :func:`~OpenPinch.main.pinch_analysis_service`.
2. Load a problem file into
   :class:`~OpenPinch.classes.pinch_problem.PinchProblem`, call
   :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.run`, and inspect or
   export the cached results.

The lower-level helpers documented on this page are still useful when you want
to separate validation, preparation, targeting, and result extraction into
distinct steps.

Service-Layer Example
---------------------

.. code-block:: python

   from OpenPinch import pinch_analysis_service
   from OpenPinch.lib.schema import StreamSchema, TargetInput

   payload = TargetInput(
       streams=[
           StreamSchema(
               zone="Process",
               name="Hot Feed",
               t_supply=180.0,
               t_target=80.0,
               heat_flow=2500.0,
               dt_cont=10.0,
           )
       ],
       utilities=[],
   )

   result = pinch_analysis_service(payload, project_name="Example")

Top-Level Package Re-Exports
----------------------------

The :mod:`OpenPinch` package re-exports the small subset of classes and helpers
that are intended to be imported most often. This makes it practical to work
from ``from OpenPinch import ...`` in notebooks and scripts without traversing
the full package layout.

In particular, the package root exposes:

- :class:`~OpenPinch.classes.pinch_problem.PinchProblem`
- :func:`~OpenPinch.main.pinch_analysis_service`
- :func:`~OpenPinch.main.get_targets`
- :func:`~OpenPinch.main.extract_results`
- :func:`~OpenPinch.utils.stream_linearisation.get_piecewise_linearisation_for_streams`

Package Entrypoints
-------------------

.. automodule:: OpenPinch
   :members:

Core Service Functions
----------------------

:mod:`OpenPinch.main` is the thin orchestration layer above the analysis
modules.

- :func:`~OpenPinch.main.pinch_analysis_service` validates the incoming payload,
  prepares the zone hierarchy, runs the appropriate direct and indirect
  targeting steps, and returns a structured response.
- :func:`~OpenPinch.main.get_targets` accepts an already prepared
  :class:`~OpenPinch.classes.zone.Zone` tree and dispatches it to the correct
  zone-level targeting routine.
- :func:`~OpenPinch.main.extract_results` converts the solved in-memory zone
  hierarchy into the dictionary structure consumed by
  :class:`~OpenPinch.lib.schema.TargetOutput`.

.. automodule:: OpenPinch.main
   :members:

PinchProblem Convenience Wrapper
--------------------------------

:class:`~OpenPinch.classes.pinch_problem.PinchProblem` adds file loading,
cached execution state, tabular summaries, graph generation, Excel export, and
Streamlit dashboard integration on top of the core service layer.

Use it when you want:

- a single object that owns the problem definition and solved result
- support for JSON, workbook, and CSV-bundle inputs
- simple summary, graph, export, and dashboard hooks without manually wiring the lower-level
  functions

The main user-facing methods are:

- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.run`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.validate`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.summary_frame`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.plot_composite_curve`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.plot_grand_composite_curve`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.compare_to`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.evaluate_heat_pump_integration`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.export_graphs`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.export_excel`
- :meth:`~OpenPinch.classes.pinch_problem.PinchProblem.show_dashboard`

The wrapper is intentionally light. Once targeting has run, the same solved
:class:`~OpenPinch.classes.zone.Zone` hierarchy and
:class:`~OpenPinch.lib.schema.TargetOutput` objects remain available for direct
inspection.

.. automodule:: OpenPinch.classes.pinch_problem
   :members:
