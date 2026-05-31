PinchWorkspace
==============

:class:`OpenPinch.classes.pinch_workspace.PinchWorkspace` is the public
multi-case interface in OpenPinch. It keeps named study cases, but each case
still resolves to a real :class:`~OpenPinch.classes.pinch_problem.PinchProblem`
instance for targeting, plotting, summaries, export, and advanced workflows.

When To Use It
--------------

Use ``PinchWorkspace`` when you want:

- named baseline-versus-variant study cases
- script or notebook comparisons without rebuilding case-input helpers
- case copying, input editing, and bundle save/load in one object
- the normal ``PinchProblem`` workflow on each case once it is selected

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

Core Workflow Members
---------------------

The main user-facing workflow members are ``case()``, ``use_case()``,
``copy_case()``, ``list_cases()``, ``get_case_payload()``,
``update_options()``, ``set_dt_cont_multiplier()``, ``compare_cases()``,
``solve_variant()``, ``compare_variants()``, ``save_bundle()``, and
``load_bundle()``.

PinchWorkspace API
------------------

.. autoclass:: OpenPinch.classes.pinch_workspace.PinchWorkspace
   :members:
   :no-index:

Relationship To PinchProblem
----------------------------

``PinchWorkspace`` is not a second solver. It orchestrates multiple named
case inputs and live ``PinchProblem`` cases on top of the same validation,
preparation, targeting, graph, and export surfaces documented for
``PinchProblem``.

Use :doc:`pinchproblem` when you only need one case at a time. Use
``PinchWorkspace`` when the study itself has to remember multiple named cases.
