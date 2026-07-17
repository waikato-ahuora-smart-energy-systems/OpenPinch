PinchWorkspace API
==================

``PinchWorkspace`` is the high-level multi-case coordinator. Import it with
``from OpenPinch import PinchWorkspace``. It keeps named study cases, but each case
still resolves to a real :class:`~OpenPinch.application.problem.PinchProblem`
instance for targeting, plotting, summaries, export, and advanced workflows.
It also exposes a second surface for serializable variant views that are useful
for applications or frontend-oriented comparisons.

When To Use It
--------------

Use ``PinchWorkspace`` when you want:

- named baseline-versus-variant study cases
- script or notebook comparisons without rebuilding case-input helpers
- case copying, input editing, and bundle save/load in one object
- the normal ``PinchProblem`` workflow on each case once it is selected
- serializable comparison views in addition to live ``PinchProblem`` objects

Lifecycle
---------

The typical lifecycle is:

1. construct the workspace from input data, an existing ``PinchProblem``, or a
   packaged sample-case name
2. use ``case(...)`` or ``use_case(...)`` to work with the active
   ``PinchProblem``
3. clone cases with ``copy_case(...)`` and mutate them with
   ``update_options(...)`` or ``set_dt_cont_multiplier(...)``
4. compare cases with ``compare_cases(...)`` or solve serializable views with
   ``solve_variant(...)``
5. persist the study with ``save_bundle(...)``

Bundle Contract
---------------

Workspace bundles use schema version ``2``. Each item in ``variants`` stores
its canonical problem mapping under ``case_input``. Version ``1``, unknown
versions, the retired ``payload`` field, and unknown bundle fields are rejected;
the pre-release API does not provide a compatibility loader.

Core Workflow Members
---------------------

The main user-facing workflow members are ``case()``, ``use_case()``,
``copy_case()``, ``list_cases()``, ``get_case_input()``,
``update_options()``, ``set_dt_cont_multiplier()``, ``compare_cases()``,
``solve_variant()``, ``compare_variants()``, ``save_bundle()``, and
``load_bundle()``.

Two Related Surfaces
--------------------

Live-case surface
   ``case()``, ``use_case()``, ``target``, ``plot``, ``summary_frame()``,
   ``export_excel()``, and ``show_dashboard()`` all delegate to a real active
   :class:`~OpenPinch.application.problem.PinchProblem`.
   That live case can use ``problem.target.exergy(...)`` and the matching
   exergetic plot accessors exactly like any other ``PinchProblem``.

Serializable variant surface
   ``input_view()``, ``validate_variant()``, ``solve_variant()``, and
   ``compare_variants()`` return schema-backed views that are deterministic and
   easier to pass to another application layer.

Internal Workflow Categories
----------------------------

``solve_variant()`` accepts the same high-level workflow names exposed on
``problem.target.*`` after normalizing case, hyphens, and spaces.

Core internal workflows
   ``target``, ``direct_heat_integration``, ``indirect_heat_integration``

Advanced internal workflows
   ``direct_heat_pump``, ``indirect_heat_pump``,
   ``direct_refrigeration``, ``indirect_refrigeration``, ``cogeneration``,
   ``area_cost``

The workspace labels the second group as advanced so repository applications
can surface that distinction. Neither group is compatibility-protected.

``solve_variant()`` does not currently normalize a dedicated ``exergy``
workflow name. Use the live-case ``PinchProblem`` surface when you need the
exergy post-processing accessor.

Typical Pattern
---------------

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="crude_preheat_train.json",
       project_name="crude_preheat_train",
   )
   baseline = workspace.case("baseline")
   workspace.copy_case("baseline", "wide_dt", activate=False)
   workspace.set_dt_cont_multiplier(0.5, case_name="wide_dt")
   comparison = workspace.compare_cases("baseline", "wide_dt")

For application-style usage:

.. code-block:: python

   view = workspace.solve_variant(
       "baseline",
       workflow="indirect_heat_integration",
   )
   comparison_view = workspace.compare_variants(["baseline", "wide_dt"])

PinchWorkspace API
------------------

.. autoclass:: OpenPinch.application.workspace.PinchWorkspace
   :members:
   :no-index:

Relationship To PinchProblem
----------------------------

``PinchWorkspace`` is not a second solver. It orchestrates multiple named
case inputs and live ``PinchProblem`` cases on top of the same validation,
preparation, targeting, graph, export, and period workflow surfaces
documented for ``PinchProblem``.

Use :doc:`pinchproblem` when you only need one case at a time. Use
``PinchWorkspace`` when the study itself has to remember multiple named cases.
