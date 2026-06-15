Heat Pump Workflows
===================

OpenPinch exposes two related but distinct Heat Pump workflow families.

Question This Guide Answers
---------------------------

How do I use OpenPinch to evaluate Heat Pump or refrigeration opportunities in
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
indirect HPR comparison workflow. It keeps the study orchestration on
``PinchWorkspace`` and inspects the resulting HPR target through the standard
plot accessor surfaces, especially
``problem.plot.net_load_profiles(zone_name="Direct Heat Pump")`` and
``problem.plot.grand_composite_curve_with_heat_pump(...)``.

Current Recommendation
----------------------

For supported advanced Heat Pump and refrigeration work today, prefer the
``problem.target.direct_heat_pump(...)``,
``problem.target.indirect_heat_pump(...)``, and refrigeration companion
methods. Use ``chocolate_factory.json`` plus notebook 03 when the question is
direct-versus-indirect comparison over a study range, and use
``heat_pump_targeting.json`` when you want a smaller direct HPR screening
input data.

Start simulated-cycle studies from a Carnot solve where possible. Use
``HPR_TYPE = "Cascade Carnot cycles"`` for broad screening,
``HPR_TYPE = "Parallel Carnot cycles"`` when you want one explicit Carnot
stage per temperature pair, and then move to
``"Parallel vapour compression cycles"``,
``"Cascade vapour compression cycles"``, or
``"Vapour compression with MVR cascade"`` when refrigerant-specific behaviour
matters.

What To Compare
---------------

Start with:

- hot utility target change
- cold utility target change
- heat recovery change
- graph change, especially in the GCC

Treat cycle-level quantities as supporting context after the integration-level
answer looks promising.

For the simulated aggregate backends, optimiser variables named
``x_heat_base``/``x_cool_base`` set the total cycle scale and
``x_heat_split``/``x_cool_split`` distribute that scale between stages. The
backend classes clip requested stage duties to the process availability before
solving refrigerant or Carnot physics.

Next Steps
----------

- For the technical framing, see
  :doc:`../fundamentals/heat-pump-and-refrigeration-methods`.
- For the deeper implementation stack, see :doc:`../api/generated-index`.
