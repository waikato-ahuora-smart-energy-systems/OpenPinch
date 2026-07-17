Cogeneration Workflows
======================

.. warning::

   This advanced guide uses unsupported internal owner modules. Only
   :func:`OpenPinch.main.pinch_analysis_service` is compatibility protected.

Purpose
-------

Use cogeneration targeting when a solved thermal target should be screened for
above Pinch or below Pinch turbine work opportunities.

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

   from OpenPinch.application.problem import PinchProblem

   problem = PinchProblem("pulp_mill.json")
   problem.target.indirect_heat_integration()
   cogeneration_target = problem.target.cogeneration()
   summary = problem.summary_frame()

Expected Output
---------------

The cogeneration target adds turbine work and efficiency fields to the solved
target context. It does not replace the base thermal answer.

Interpretation
--------------

Read cogeneration results in this order:

1. base thermal target and utility levels
2. work target
3. turbine efficiency target
4. stage details

Key turbine assumptions live on ``zone.config`` and include ``TURB_T_IN``,
``TURB_P_IN``, ``MIN_EFF``, ``LOAD_FRACTION``, ``ETA_MECH``,
``TURB_MODEL``, and ``IS_HIGH_P_COND_FLASH``.

Next Steps
----------

- :doc:`../fundamentals/cogeneration-methods` for method framing.
- :doc:`../api/pinchproblem` for the internal parent accessor.
- :doc:`../api/service-layer` for lower-level orchestration.
