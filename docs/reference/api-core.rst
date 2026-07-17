Main Contract and Application Workflows
=======================================

The public workflow coordinators are imported from :mod:`OpenPinch`. The
mapping-in/result-out service remains the strict protected integration
contract. Lower-level orchestration helpers below are contributor references.

Recommended Usage
-----------------

For stateful workflow code, import ``PinchProblem`` or ``PinchWorkspace`` from
``OpenPinch``. For service-style code, pass a request mapping to
:func:`~OpenPinch.main.pinch_analysis_service`.

Load a problem file into :class:`~OpenPinch.application.problem.PinchProblem`, call
:meth:`~OpenPinch.application.problem.PinchProblem.target`, and inspect or
export the cached results. Use
:class:`~OpenPinch.application.workspace.PinchWorkspace` when the study needs
named baseline and variant cases rather than one case at a time.

The lower-level helpers documented on this page are still useful when you want
to separate validation, preparation, targeting, and result extraction into
distinct steps.

Service-Layer Example
---------------------

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service
   from OpenPinch.contracts.input import StreamSchema, TargetInput

   input_data = TargetInput(
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

   result = pinch_analysis_service(input_data, project_name="Example")

Public and Concrete Owner Imports
---------------------------------

Import the two public workflow coordinators from :mod:`OpenPinch`, the
protected service from :mod:`OpenPinch.main`, and advanced objects from their
concrete contract, domain, analysis, adapter, or presentation owners.

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

Common entry points and advanced imports include:

- :class:`~OpenPinch.application.problem.PinchProblem`
- :class:`~OpenPinch.application.workspace.PinchWorkspace`
- :func:`~OpenPinch.main.pinch_analysis_service`
- :func:`~OpenPinch.domain._stream.linearisation.get_piecewise_linearisation_for_streams`

Package Entrypoints
-------------------

.. automodule:: OpenPinch
   :no-members:

.. autofunction:: OpenPinch.main.pinch_analysis_service

.. autoclass:: OpenPinch.application.problem.PinchProblem
   :members:

.. autoclass:: OpenPinch.application.workspace.PinchWorkspace
   :members:

Core Service Functions
----------------------

:mod:`OpenPinch.main` is the thin orchestration layer above the analysis
modules.

- :func:`~OpenPinch.main.pinch_analysis_service` validates the incoming input data,
  prepares the zone hierarchy, runs the appropriate direct and indirect
  targeting steps, and returns a structured response.

Private dispatch and result-extraction helpers support the internal problem
coordinator and the protected service entry point. They are not stable user
interfaces.

.. automodule:: OpenPinch.main
   :no-members:

PinchProblem Convenience Wrapper
--------------------------------

:class:`~OpenPinch.application.problem.PinchProblem` adds file loading,
cached execution state, tabular summaries, graph generation, Excel export, and
Streamlit dashboard integration on top of the core service layer. It also owns
the ``add_component`` accessor used for memory-only process mutations such as
direct gas/vapour MVR before targets are rerun.

Those rendered graph, Excel, and dashboard surfaces require the
``openpinch[notebook]`` or ``openpinch[dashboard]`` extra.

Use it when you want:

- a single object that owns the problem definition and solved result
- support for JSON, workbook, and CSV-bundle inputs
- simple summary, graph, export, and dashboard hooks without manually wiring the lower-level
  functions

The main user-facing methods are:

- :meth:`~OpenPinch.application.problem.PinchProblem.target`
- :meth:`~OpenPinch.application.problem.PinchProblem.validate`
- :meth:`~OpenPinch.application.problem.PinchProblem.summary_frame`
- :meth:`~OpenPinch.application.problem.PinchProblem.plot.composite_curve`
- :meth:`~OpenPinch.application.problem.PinchProblem.plot.grand_composite_curve`
- :meth:`~OpenPinch.application.problem.PinchProblem.compare_to`
- :meth:`~OpenPinch.application.problem.PinchProblem.export_excel`
- :meth:`~OpenPinch.application.problem.PinchProblem.show_dashboard`
- ``problem.add_component.process_mvr(...)``

The wrapper is intentionally light. Once targeting has run, the same solved
:class:`~OpenPinch.domain.zone.Zone` hierarchy and
:class:`~OpenPinch.contracts.output.TargetOutput` objects remain available for direct
inspection.

The ``problem.target.*`` accessor is the explicit advanced post-processing
entrypoint family. Each named workflow returns the affected
:class:`~OpenPinch.domain.targets.BaseTargetModel` and refreshes cached
:class:`~OpenPinch.contracts.output.TargetOutput` results on the same
:class:`~OpenPinch.application.problem.PinchProblem` instance. Heat pump and
refrigeration targets also surface HPR summary fields such as ``hpr_cycle``,
``hpr_utility_total``, ``hpr_work``, ``hpr_external_utility``, and
``StreamCollection`` objects on ``hpr_hot_streams`` and ``hpr_cold_streams``.
Simulated-cycle targets also expose annualized cost fields:
``hpr_operating_cost``, ``hpr_capital_cost``,
``hpr_annualized_capital_cost``, ``hpr_total_annualized_cost``,
``hpr_compressor_capital_cost``, and
``hpr_heat_exchanger_capital_cost``.
For multiperiod input data, the same named entry points also accept
``period_id=...`` and the refreshed summaries, exports, and graph metadata carry
that selected period context forward. Weighted HPR summaries average operating
fields, retain maximum capital fields, and recompute total annualized cost from
weighted operating cost plus maximum annualized capital. Per-period summary
replay uses fresh copies of one baseline zone and restores the original problem
state on both success and failure.

The ``problem.add_component.*`` accessor is different: it mutates the prepared
stream model before targeting. The direct process MVR component deactivates
selected source streams, activates generated compressed-vapour replacement
streams, stores per-period stage results, and contributes process-component
work to later target summaries.
