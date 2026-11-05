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

Turbine Models
--------------

``cogeneration`` uses Medina-Flores et al. (2010). ``sun_smith_cogeneration``,
``varbanov_small_turbine_cogeneration``, ``varbanov_large_turbine_cogeneration``
and ``isentropic_cogeneration`` select the other models.

The two Varbanov methods each use one size band of the Varbanov et al. (2004)
coefficients: the small model for stage work at or below 2000 kW, and the large
model for stage work above 2000 kW. Neither switches band, so choose by the size
of the turbine. Both change from a back-pressure to a condensing turbine when the
exit would be wet.

If a stage's work is outside the band, the target still reports the number. One
message, naming the stages, is added to ``turbine_diagnostics``, and this alone
does not set ``turbine_converged`` to ``False``. The test uses the steam's work
before ``POWER_ETA_MECH`` is applied.

The two models can give very different work at the same duty, so for some duties
neither is in range: the small model predicts more than 2000 kW and the large
model 2000 kW or less. A train whose stages lie either side of 2000 kW also warns
under both. Treat such a target as an extrapolation.

Convergence
-----------

``turbine_converged`` on the target and on its result row is ``False`` when the
extraction mass balance did not settle within its iteration limit, and
``turbine_diagnostics`` holds the messages. The same messages are raised as a
``UserWarning`` and listed in ``problem.report().warnings``. The work target of
such a run is its last iteration, so treat it as unreliable: a small change in
duty or in the turbine model usually restores convergence. For the Varbanov
models this can happen in a narrow range of duty around the change to a
condensing turbine, where the condensing branch does less work. A weighted-average
row is flagged if any period did not converge.

Interpretation
--------------

Read cogeneration results in this order:

1. base thermal target and utility levels
2. work target
3. turbine efficiency target
4. stage details

Stage details are in ``turbine_result`` on the target and on its result row: a
``TurbineSolveResult`` whose ``stages`` holds one ``TurbineStageResult`` for each
extraction level, hottest first. A stage's ``source_index`` is its position in
the target's ``hot_utilities``. These are raw floats in fixed units, whatever
display units are configured: heat flow and work in kW, temperature in degC,
pressure in bar, specific enthalpy in kJ/kg and mass flow in kg/s. In a
multiperiod problem each period row has its own record. The weighted-average row
has none, because an average is not a solved operating point.

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
