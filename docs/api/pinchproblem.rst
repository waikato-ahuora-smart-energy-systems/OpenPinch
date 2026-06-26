PinchProblem
============

:class:`OpenPinch.classes.pinch_problem.PinchProblem` is the primary
single-case interface in the package. It owns the source inputs, validated
problem data, prepared zone tree, solved targets, graph exports, multiperiod
reruns, and several advanced post-processing helpers.

When To Use It
--------------

Use ``PinchProblem`` when you want:

- support for JSON, workbook, CSV-bundle, packaged sample-case, or in-memory
  inputs
- one object that keeps both the original case and the solved result state
- compact and detailed summary tables
- graph building and graph export without manually wiring result payloads
- notebook-friendly advanced workflows such as HPR screening or
  ``dt_cont`` sensitivity studies
- selected-period reruns through ``period_id`` and batch reruns through
  ``target_all_periods()``

Use :doc:`pinchworkspace` instead when the study itself needs to keep multiple
named cases and compare them over time.

Supported Sources
-----------------

``PinchProblem`` accepts these load shapes:

- ``TargetInput`` or plain mappings
- JSON files
- Excel files such as ``.xlsx`` and ``.xlsb``
- a directory with ``streams.csv`` and ``utilities.csv``
- a ``(streams_csv, utilities_csv)`` tuple
- a packaged sample-case name such as ``basic_pinch.json`` when no local file
  with that name exists

Lifecycle
---------

The typical lifecycle is:

1. construct the wrapper with a project name or input source
2. call :meth:`load` or pass the source at construction time
3. call :meth:`validate` if you want a preflight check
4. call :meth:`target`
5. inspect summaries, graphs, exports, period-specific reruns, or the prepared
   ``master_zone``

When the source is a bare ``*.json`` filename and no local file exists,
``PinchProblem`` also resolves packaged sample cases such as
``basic_pinch.json`` or ``crude_preheat_train.json`` directly.

Core Workflow Members
---------------------

The main user-facing workflow members are ``load()``, ``validate()``,
``target()``, ``summary_frame()``, ``export_excel()``, ``compare_to()``,
``set_dt_cont_multiplier()``, ``update_options()``, ``target_all_periods()``,
and ``show_dashboard()``.

Period Workflows
------------------

When the prepared data contains multiple periods, the main period surfaces are:

- ``period_ids`` for the canonical ``period_id -> idx`` mapping
- named ``problem.target.*(..., period_id="peak")`` reruns
- ``target_all_periods(parallel=False | True | "thread" | "process")`` for
  batch solves across every canonical period

Period selection happens at targeting time, not at load time. The cached result,
summary frame, export surface, and graph payload then reflect the selected
state.

Advanced Entry Points
---------------------

Two descriptor families make ``PinchProblem`` broader than a simple wrapper:

``problem.plot``
   Builds Plotly figures for composite curves, grand composite curves, net-load
   profiles, exergetic post-processing graphs, and related graph families from
   the cached solved state.

``problem.target``
   Re-runs targeted advanced routines such as direct and indirect heat pump,
   refrigeration, exergy enrichment, cogeneration, or area/cost targeting
   against the prepared zone hierarchy. Named target workflows also accept
   ``period_id=...`` when input data carries multiperiod values, and the
   refreshed summary/export surfaces then expose that selected period on the
   result rows.

Named target methods also accept ``zone_name=...`` and
``include_subzones=True`` when you want one zone-level or subtree-level rerun
instead of the default root-case solve.

``problem.target.exergy(...)`` is intentionally a post-processing accessor: it
enriches one existing compatible target family instead of solving a new target
family of its own.

``problem.add_component``
   Mutates the prepared process model with memory-only process components
   before rerunning targets. The current public component surface is
   ``problem.add_component.process_mvr(...)`` for direct gas/vapour MVR. It
   deactivates selected original hot streams, activates generated replacement
   streams, and carries the process-component work into later summaries.

Those accessors are the high-level path into the package's deeper analytical
power without dropping all the way to the raw service modules.

Output and Inspection Surfaces
------------------------------

After solving, the main read and export surfaces are:

- ``summary_frame()`` for compact or detailed pandas views
- ``problem.plot.catalog()`` for the available graph inventory
- ``problem.plot.*`` for Plotly figures and raw graph payloads
- ``problem.plot.export(...)`` for standalone HTML graph files
- ``export_excel(...)`` for workbook output
- ``show_dashboard()`` for the Streamlit-based review surface

PinchProblem API
----------------

.. autoclass:: OpenPinch.classes.pinch_problem.PinchProblem
   :members:
   :no-index:

Heat Pump Integration Comparison
--------------------------------

Older examples sometimes refer to broader helper-backed HPR comparison flows.
The current object surface instead exposes the explicit advanced HPR routines
on ``problem.target.*``, described above and in :doc:`service-layer`.

Relationship To Lower Layers
----------------------------

``PinchProblem`` is a wrapper, not a separate solver. Under the hood it still
validates the input data, prepares a :class:`~OpenPinch.classes.zone.Zone` tree,
runs the same targeting services documented in :doc:`service-layer`, and then
packages the outputs for summaries, graphs, and export.

If you need to mutate the prepared zone tree directly, inspect stream objects,
or run only one analysis stage, move down to :doc:`service-layer` and
:doc:`domain-model`.
