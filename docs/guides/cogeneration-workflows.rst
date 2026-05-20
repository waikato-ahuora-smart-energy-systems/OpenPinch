Cogeneration Workflows
======================

OpenPinch exposes cogeneration as an advanced post-processing workflow on top
of solved thermal targets.

Question This Guide Answers
---------------------------

How do I screen turbine cogeneration opportunities from a solved OpenPinch
case?

Typical Workflow
----------------

1. solve the base thermal case
2. inspect the hot utility picture
3. run the cogeneration workflow on the relevant target
4. compare work and efficiency targets with the underlying utility structure

Python Surface
--------------

The main user-facing route is:

.. code-block:: python

   problem = PinchProblem("pulp_mill.json")
   problem.target()
   cogeneration_target = problem.target.cogeneration()

Configuration
-------------

The turbine parameters are part of `zone.config`, including:

- `TURB_T_IN`
- `TURB_P_IN`
- `MIN_EFF`
- `LOAD_FRACTION`
- `ETA_MECH`
- `TURB_MODEL`
- `IS_HIGH_P_COND_FLASH`

This makes cogeneration studies consistent with the rest of the package model:
runtime assumptions are attached to the zone hierarchy rather than hidden in a
side channel.

How To Interpret The Result
---------------------------

Use cogeneration outputs after the thermal answer is already understood.

Read them in this order:

1. utility structure and thermal target context
2. work target
3. efficiency target
4. stage detail

Useful Sample
-------------

The `pulp_mill.json` and `zonal_site.json` assets are good next cases once the
basic process-level workflow is understood, especially when combined with the
packaged Total Site notebook.

Next Steps
----------

- For the method framing, see :doc:`../fundamentals/cogeneration-methods`.
- For the exact API surfaces, see :doc:`../api/pinchproblem`.
