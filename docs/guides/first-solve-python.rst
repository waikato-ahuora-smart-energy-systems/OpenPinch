First Solve with Python
=======================

This is the primary OpenPinch workflow for a process engineer.

.. code-block:: python

   from OpenPinch import PinchProblem

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

   problem = PinchProblem(request, project_name="First solve")
   problem.validate()
   problem.target.all_heat_integration()

   summary = problem.summary_frame()
   metrics = problem.metrics()
   report = problem.report()
   gcc = problem.plot.grand_composite_curve()

The lifecycle is explicit. Construction prepares the case,
``all_heat_integration()`` performs dependency-aware analysis, and subsequent
operations observe the cached result. Invalid input raises a Pydantic
validation error before analysis begins.

Use a focused method when appropriate:

.. code-block:: python

   direct = problem.target.direct_heat_integration()
   area_cost = problem.target.heat_exchanger_area_and_cost()

Method arguments are ephemeral. Persistent engineering fallbacks belong in
``problem.update_options(...)``; configuration does not decide which method
runs.

Next Steps
----------

- :doc:`zonal-and-total-site-workflows`
- :doc:`heat-pump-workflows`
- :doc:`../examples/notebook-series`
- :doc:`../api/pinchproblem`
