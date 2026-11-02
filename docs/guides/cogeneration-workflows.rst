Cogeneration Workflows
======================

Purpose
-------

Use cogeneration targeting when a solved thermal target should be screened for
above Pinch turbine work opportunities. The public workflow runs the above
Pinch screening only.

Prerequisites
-------------

Solve the base thermal case first. Cogeneration is an advanced
post-processing workflow; it should be interpreted after the utility structure
is understood.

Sample Case
-----------

Use ``pulp_mill.json`` for Total Site and turbine screening examples.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("pulp_mill.json")
   problem.target.all_heat_integration()
   base = problem.target.total_site_heat_integration()
   cogeneration_target = problem.target.cogeneration(base_target=base)
   cogeneration_target.work_target
   cogeneration_target.turbine_efficiency_target

Expected Output
---------------

The cogeneration target adds turbine work and efficiency fields. Read them on
the returned target or in ``problem.results.targets``; ``summary_frame()`` has
no work or efficiency column. The cogeneration target does not replace the base
thermal answer.

Convergence
-----------

``turbine_converged`` on the target and on its result row is ``False`` when the
extraction mass balance did not settle within its iteration limit, and
``turbine_diagnostics`` holds the messages. The same messages are raised as a
``UserWarning`` and listed in ``problem.report().warnings``. The work target of
such a run is its last iteration, so treat it as unreliable: a small change in
duty or in the turbine model usually restores convergence. A weighted-average
row is flagged if any period did not converge.

Interpretation
--------------

Read cogeneration results in this order:

1. base thermal target and utility levels
2. work target
3. turbine efficiency target

Stage details are not stored on the result.

Key turbine assumptions are the option keys ``POWER_TURB_T_IN``,
``POWER_TURB_P_IN``, ``POWER_MIN_EFF``, ``POWER_LOAD_FRACTION``,
``POWER_ETA_MECH``, and ``POWER_HIGH_P_COND_FLASH_ENABLED``. They are read at
runtime from ``zone.config.power`` (for example ``zone.config.power.turb_t_in``).
The turbine model is selected by the target method name, not by an option.

Next Steps
----------

- :doc:`../fundamentals/cogeneration-methods` for method framing.
- :doc:`../api/pinchproblem` for the internal parent accessor.
- :doc:`../api/service-layer` for lower-level orchestration.
