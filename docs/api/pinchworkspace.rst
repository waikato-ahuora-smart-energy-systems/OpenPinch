PinchWorkspace
==============

:class:`OpenPinch.PinchWorkspace` manages named :class:`OpenPinch.PinchProblem`
cases. The public vocabulary is consistently case-oriented.

Golden Path
-----------

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace("crude_preheat_train.json", project_name="Site")
   baseline = workspace.case("baseline")
   baseline.target.direct_heat_integration()

   retrofit = workspace.scenario("retrofit", dt_cont_multiplier=0.8)
   retrofit.target.direct_heat_integration()

   comparison = workspace.compare_cases("baseline", "retrofit")

Interaction Matrix
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 29 19 17 10

   * - Surface
     - Purpose
     - Return
     - State effect
     - Dependency
   * - construction, ``load``, ``case``, ``use_case``, ``list_cases``
     - Load, retrieve, select, and list named cases
     - problem or names
     - prepare/select
     - base
   * - ``scenario``
     - Create an unsolved case from an existing case with named overrides
     - problem
     - add case
     - base
   * - ``cases(names).target.*`` and ``cases(names).design.*``
     - Run one named workflow across an ordered case set
     - ordered case result
     - per-case
     - method-specific
   * - ``cases(names).summary_frames()``, ``metrics()``, and ``reports()``
     - Observe every selected solved case and preserve per-case failures
     - result and error mappings
     - none
     - base
   * - ``cases(names).export_excel(directory)``
     - Export each case workbook into a distinctly named case subdirectory
     - result and error mappings
     - explicit files
     - excel
   * - ``compare_cases`` and ``compare_to``
     - Compare cached numerical summaries
     - dataframe
     - none
     - base
   * - ``target``, ``design``, ``components``, ``plot``, ``config``
     - Forward to the active case
     - active-case accessor
     - method-specific
     - method-specific
   * - ``update_options`` and ``set_dt_cont_multiplier``
     - Change persistent fallback values on a named or active case
     - none
     - invalidates case
     - base
   * - ``summary_frame``, ``metrics``, ``report``, validation and state
       properties
     - Observe the selected case without solving
     - dataframe, mapping, report, or record
     - none
     - base
   * - ``save_bundle``, ``load_bundle``, ``to_problem_json``
     - Persist or recover case inputs
     - path, workspace, or mapping
     - explicit persistence
     - base
   * - ``export_excel`` and ``show_dashboard``
     - Explicit output side effects for the active case
     - path or dashboard handle
     - none
     - output-specific

Observation methods do not solve an unsolved scenario. Call the desired
``target`` or ``design`` method first. Method arguments remain ephemeral;
``update_options(...)`` changes stored fallbacks.

Batch target and design namespaces are intentionally separate: a target batch
does not advertise HEN design methods, and a design batch does not advertise
targeting methods. One case failing does not discard successful cases; inspect
``outcome.errors`` before using ``outcome.results``.

Complete API
------------

.. autoclass:: OpenPinch.PinchWorkspace
   :members:
   :undoc-members:

See :doc:`../examples/tutorial-coverage-map` for every workspace operation and
its executable tutorial owner.
