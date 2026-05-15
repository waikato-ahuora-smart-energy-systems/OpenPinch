Heat Pump Workflows
===================

OpenPinch exposes two related but distinct heat-pump workflow families.

Question This Guide Answers
---------------------------

How do I use OpenPinch to evaluate heat-pump or refrigeration opportunities in
the context of process integration?

Direct and Indirect HPR Targeting
---------------------------------

Use this when you want the package to compute or screen HPR-oriented targets
more directly.

Typical surfaces:

- `problem.target.direct_heat_pump(...)`
- `problem.target.indirect_heat_pump(...)`
- `problem.target.direct_refrigeration(...)`
- `problem.target.indirect_refrigeration(...)`

Use this workflow when your main question is:

- how do direct and indirect HPR routes compare over a study range?

Recommended Learning Asset
--------------------------

The packaged notebook:

.. code-block:: bash

   openpinch notebook --name 03_carnot_hpr_comparison.ipynb -o notebooks

This is the best single packaged learning asset for the broader direct-versus-
indirect HPR comparison workflow.

Current Recommendation
----------------------

For supported advanced heat-pump and refrigeration work today, prefer the
``problem.target.direct_heat_pump(...)``,
``problem.target.indirect_heat_pump(...)``, and refrigeration companion
methods. Use ``chocolate_factory.json`` plus notebook 03 when the question is
direct-versus-indirect comparison over a study range, and use
``heat_pump_targeting.json`` when you want a smaller direct HPR screening
payload.

What To Compare
---------------

Start with:

- hot utility target change
- cold utility target change
- heat recovery change
- graph change, especially in the GCC

Treat cycle-level quantities as supporting context after the integration-level
answer looks promising.

Next Steps
----------

- For the technical framing, see
  :doc:`../fundamentals/heat-pump-and-refrigeration-methods`.
- For the deeper implementation stack, see :doc:`../api/generated-index`.
