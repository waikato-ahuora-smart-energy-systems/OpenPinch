Quickstart Workflow
===================

This walkthrough shows how to describe a simple process, run the targeting
workflow, and inspect the results programmatically.

Step 1. Describe Your Streams and Utilities
-------------------------------------------

Targeting requires structured descriptions of hot/cold process streams and
candidate utilities.  The schema objects exported from
:mod:`OpenPinch.lib.schema` validate data before execution.

.. code-block:: python

   from OpenPinch.lib.schema import StreamSchema, UtilitySchema, TargetInput
   from OpenPinch.lib.enums import StreamType

   streams = [
       StreamSchema(
           zone="Process Unit",
           name="Reboiler Vapor",
           t_supply=200.0,
           t_target=120.0,
           heat_flow=8_000.0,
           dt_cont=10.0,
           htc=1.5,
       ),
       StreamSchema(
           zone="Process Unit",
           name="Feed Preheat",
           t_supply=40.0,
           t_target=160.0,
           heat_flow=6_000.0,
           dt_cont=10.0,
           htc=1.2,
       ),
   ]

   utilities = [
       UtilitySchema(
           name="Cooling Water",
           type=StreamType.Cold,
           t_supply=25.0,
           t_target=35.0,
           heat_flow=120_000.0,
           dt_cont=5.0,
           htc=0.8,
           price=12.0,
       )
   ]

   payload = TargetInput(streams=streams, utilities=utilities)

Step 2. Run the Service
-----------------------

Use :func:`OpenPinch.pinch_analysis_service` to run the pipeline end-to-end.  The
service accepts dictionaries, Pydantic models, or dataclass-like objects and
returns :class:`OpenPinch.lib.schema.TargetOutput`.

.. code-block:: python

   from OpenPinch import pinch_analysis_service

   result = pinch_analysis_service(payload)
   for target in result.targets:
       print(target.name, target.Qh, target.Qc)

Step 3. Persist or Post-process
-------------------------------

For convenience you can wrap the pipeline in :class:`OpenPinch.PinchProblem`.
It handles loading from JSON/Excel/CSV bundles and exporting the results to a
format compatible with the legacy Excel workbook.

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem(problem_filepath="sample_problem.json", run=True)
   problem.export("results/")

Next Steps
----------

- Explore :mod:`OpenPinch.analysis` for lower-level building blocks.
- Use the schema models under :mod:`OpenPinch.lib.schema` to validate larger data sets.
- Continue to :doc:`../reference/index` for the API reference.
