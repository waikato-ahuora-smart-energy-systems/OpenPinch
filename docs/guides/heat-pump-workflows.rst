Heat Pump Workflows
===================

OpenPinch exposes two related but distinct Heat Pump workflow families.
It also includes a direct process MVR component path for cases where a hot
gas/vapour stream is recompressed before the normal direct or Total Site
targeting workflow.

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

Direct Gas/Vapour MVR Components
--------------------------------

Use this when the process stream itself is the MVR source. Unlike the HPR
targeting routines above, direct process MVR mutates a prepared
``PinchProblem`` by deactivating selected original hot streams and activating
replacement hot streams generated from the compressed vapour cooling profile.
The subsequent direct or Total Site target then includes the component work in
the solved summary.

Typical surface:

- ``problem.add_component.process_mvr(...)``

The packaged notebook:

.. code-block:: bash

   openpinch notebook --name 08_direct_gas_stream_mvr.ipynb -o notebooks

This notebook compares baseline, dry MVR, and liquid-injection MVR cases in a
``PinchWorkspace``. It also shows ``stage_results_by_state``, replacement
stream inspection, and component ``activate()`` / ``deactivate()`` behavior.

Current Recommendation
----------------------

For supported advanced Heat Pump and refrigeration work today, prefer the
``problem.target.direct_heat_pump(...)``,
``problem.target.indirect_heat_pump(...)``, and refrigeration companion
methods. Use ``chocolate_factory.json`` plus notebook 03 when the question is
direct-versus-indirect comparison over a study range, and use
``heat_pump_targeting.json`` when you want a smaller direct HPR screening
input data. Use notebook 08 when the question is direct recompression of an
existing process gas/vapour stream before re-solving the base integration
targets.

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

- total annualized HPR cost for simulated-cycle backends
- hot utility target change
- cold utility target change
- heat recovery change
- graph change, especially in the GCC

Treat cycle-level quantities as supporting context after the integration-level
answer looks promising.

For ``"Parallel vapour compression cycles"``,
``"Cascade vapour compression cycles"``, and
``"Vapour compression with MVR cascade"``, OpenPinch reports unit-aware HPR
cost fields:

- ``hpr_operating_cost`` in ``$/y``
- ``hpr_capital_cost`` in ``$``
- ``hpr_annualized_capital_cost`` in ``$/y``
- ``hpr_total_annualized_cost`` in ``$/y``
- ``hpr_compressor_capital_cost`` in ``$``
- ``hpr_heat_exchanger_capital_cost`` in ``$``

The simulated-cycle objective minimises ``hpr_total_annualized_cost`` plus
feasibility penalties. Remaining external utility at the ends of the combined
residual GCC is costed as operating cost. Residual GCC pockets, opposite
utility regression, and cycle allocation penalties are the feasibility terms.

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
