PinchProblem
============

:class:`OpenPinch.classes.pinch_problem.PinchProblem` is the primary
single-case interface in the package. It owns the source inputs, validated
problem data, prepared zone tree, solved targets, graph exports, and several
advanced post-processing helpers.

When To Use It
--------------

Use ``PinchProblem`` when you want:

- support for JSON, workbook, or CSV-bundle inputs
- one object that keeps both the original case and the solved result state
- compact and detailed summary tables
- graph building and graph export without manually wiring result payloads
- notebook-friendly advanced workflows such as HPR screening or
  ``dt_cont`` sensitivity studies

Use :doc:`pinchworkspace` instead when the study itself needs to keep multiple
named cases and compare them over time.

Lifecycle
---------

The typical lifecycle is:

1. construct the wrapper with a project name or input source
2. call :meth:`load` or pass the source at construction time
3. call :meth:`validate` if you want a preflight check
4. call :meth:`target`
5. inspect summaries, graphs, exports, or the prepared ``master_zone``

When the source is a bare ``*.json`` filename and no local file exists,
``PinchProblem`` also resolves packaged sample cases such as
``basic_pinch.json`` or ``crude_preheat_train.json`` directly.

Core Workflow Members
---------------------

The main user-facing workflow members are ``load()``, ``validate()``,
``target()``, ``summary_frame()``, ``export_excel()``, ``compare_to()``,
``set_dt_cont_multiplier()``, and ``show_dashboard()``.

Advanced Entry Points
---------------------

Two descriptor families make ``PinchProblem`` broader than a simple wrapper:

``problem.plot``
   Builds Plotly figures for composite curves, grand composite curves, net-load
   profiles, and related graph families from the cached solved state.

``problem.target``
   Re-runs targeted advanced routines such as direct and indirect heat pump,
   refrigeration, cogeneration, or area/cost targeting against the prepared
   zone hierarchy. Named target workflows also accept ``state_id=...`` when a
   input data carries stateful values, and the refreshed summary/export surfaces
   then expose that selected state on the result rows.

Those accessors are the high-level path into the package's deeper analytical
power without dropping all the way to the raw service modules.

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
