First Solve with Python
=======================

The main Python single-case front door is :class:`OpenPinch.PinchProblem`.

Question This Guide Answers
---------------------------

How do I run a supported end-to-end OpenPinch workflow from a script or
notebook and inspect the results directly in Python?

Step 1. Load a Known-Good Sample
--------------------------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="basic_pinch")

Step 2. Validate and Run
------------------------

.. code-block:: python

   problem.validate()
   result = problem.target()

The `target()` call performs the high-level targeting workflow and caches the
structured result on the same object.

Step 3. Read the Summary
------------------------

.. code-block:: python

   summary = problem.summary_frame()
   print(summary)

Read the main process or plant row in this order:

1. `Hot Utility Target`
2. `Cold Utility Target`
3. `Heat Recovery`
4. `Hot Pinch` and `Cold Pinch`

Step 4. Inspect Graphs
----------------------

.. code-block:: python

   gcc = problem.plot.grand_composite_curve()
   cc = problem.plot.composite_curve()

Use the GCC first when utility placement or Heat Pump opportunity is the main
question. These rendered Plotly figures require the ``openpinch[notebook]`` or
``openpinch[dashboard]`` extra.

Step 5. Export Artifacts
------------------------

.. code-block:: python

   workbook_path = problem.export_excel("results")
   graph_paths = problem.plot.export("graphs", graph_type="gcc")

These Excel and Plotly export hooks require the ``openpinch[notebook]`` or
``openpinch[dashboard]`` extra.

Step 6. Use the Richer Workflow Hooks
-------------------------------------

`PinchProblem` also exposes:

- `problem.target.direct_heat_integration(...)`
- `problem.target.indirect_heat_integration(...)`
- `problem.target.direct_heat_pump(...)`
- `problem.target.indirect_heat_pump(...)`
- `problem.target.cogeneration(...)`
- `problem.target.direct_heat_pump(...)` and
  `problem.target.indirect_heat_pump(...)`

These are best treated as explicit advanced workflows after you understand the
base case.

Named Multi-Case Alternative
----------------------------

When you want one notebook or script to keep named study cases together, use
``PinchWorkspace``:

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

Schema-First Alternative
------------------------

When you do not want a stateful wrapper object, use the service boundary:

.. code-block:: python

   from OpenPinch import pinch_analysis_service
   from OpenPinch.lib.schemas.io import StreamSchema, TargetInput

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
       ]
   )

   result = pinch_analysis_service(payload, project_name="Example")

When To Drop Lower
------------------

Use the lower-level service and prepared-zone workflow only when you need to
inspect or mutate the intermediate `Zone` hierarchy directly.

Next Steps
----------

- For input modeling guidance, see :doc:`input-formats-and-validation`.
- For named study-case orchestration, see :doc:`../api/pinchworkspace`.
- For zonal and site workflows, see :doc:`zonal-and-total-site-workflows`.
- For the exact `PinchProblem` API, see :doc:`../api/pinchproblem`.
