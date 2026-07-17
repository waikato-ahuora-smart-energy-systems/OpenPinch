First Solve with Python
=======================

Purpose
-------

Call :func:`OpenPinch.main.pinch_analysis_service`, the sole current external
Python contract, and inspect its structured result.

Prerequisites
-------------

Install the base package:

.. code-block:: bash

   python -m pip install openpinch

Sample Case
-----------

The example uses two inline process streams so it is independent of repository
paths and resource-helper imports.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

   request = {
       "streams": [
           {
               "name": "Hot feed",
               "zone": "Process",
               "t_supply": 180.0,
               "t_target": 80.0,
               "heat_flow": 1000.0,
           },
           {
               "name": "Cold feed",
               "zone": "Process",
               "t_supply": 20.0,
               "t_target": 120.0,
               "heat_flow": 800.0,
           },
       ],
       "utilities": [],
   }

   result = pinch_analysis_service(request, project_name="first-solve")
   output = result.model_dump(mode="json")
   print(output["name"])
   print(output["targets"])

Expected Output
---------------

``output`` has the stable top-level field order ``name``, ``period_id``,
``targets``, ``graphs``, and ``design``. Invalid input raises Pydantic's
validation error before targeting begins.

Interpretation
--------------

Read each target's ``Qh``, ``Qc``, and ``Qr`` fields as the hot-utility,
cold-utility, and heat-recovery targets. Values retain their units in the
serialized structure.

Advanced Internal Workflows
---------------------------

Packaged notebooks also exercise ``PinchProblem``, ``PinchWorkspace``, HPR,
HEN, plotting, and export modules. Those concrete-owner imports are useful for
development and research, but are deliberately unsupported as external
contracts in version 0.5.0.

Next Steps
----------

- :doc:`../api/package-root` for the support boundary.
- :doc:`input-formats-and-validation` for request-field details.
- :doc:`../developer/architecture` for internal owner responsibilities.
- :doc:`notebooks-and-sample-cases` for advanced unsupported examples.
