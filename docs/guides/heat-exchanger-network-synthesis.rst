Heat Exchanger Network Synthesis
================================

.. warning::

   HEN synthesis is an advanced unsupported internal workflow. Only
   :func:`OpenPinch.main.pinch_analysis_service` is compatibility protected.

Purpose
-------

Use heat exchanger network synthesis when a solved OpenPinch case should be
converted into ranked network candidates and grid-diagram views.

Prerequisites
-------------

Install the synthesis extra and IDAES solver extensions before running
solver-backed HEN synthesis:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"
   idaes get-extensions

Source OpenHENS CSV files are migration source material only. Convert them
once into OpenPinch JSON or native ``TargetInput`` models before synthesis.

Sample Case
-----------

Use ``Four-stream-Yee-and-Grossmann-1990-1.json`` for the compact converted
OpenHENS benchmark used in the packaged HEN notebook.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch.application.problem import PinchProblem
   from OpenPinch.domain.enums import HENDesignMethod
   from OpenPinch.presentation.network_grid.service import build_grid_diagram

   problem = PinchProblem(
       "Four-stream-Yee-and-Grossmann-1990-1.json",
       project_name="Four-stream converted OpenHENS example",
   )

   design = problem.design.enhanced_synthesis_method(quality_tier=2)
   network = design.network
   period_id = network.period_ids[0]
   diagram = build_grid_diagram(network, period_id=period_id)

Explicit design-method accessors are also available:

.. code-block:: python

   problem.design.open_hens_method()
   problem.design.heat_exchanger_network_synthesis()
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.OpenHENS,
   )
   problem.design.pinch_design_method()
   problem.design.thermal_derivative_method(initial_networks=(seed_network,))
   problem.design.network_evolution_method(initial_networks=(existing_network,))

Expected Output
---------------

Successful synthesis stores a design result on ``TargetOutput.design`` and
``problem.results.design``. Inspect:

- ``design.design_method`` for the requested internal design service
- ``design.manifest.method_sequence`` for executed task-level methods
- ``design.network`` for the selected network
- ``design.ranked_networks`` for ranked unique candidates
- ``design.network.exchangers[0].state(period_id)`` for one match's operating
  duty, activity, approaches, split fractions, and temperatures
- ``design.network.total_duty(period_id=period_id)`` for period duty totals
- ``build_grid_diagram(design.network, period_id=period_id)`` for visual
  topology inspection

Interpretation
--------------

The internal design accessor is problem-rooted. Persistent synthesis controls
belong in loaded ``TargetInput.options`` keys such as ``HENS_APPROACH_TEMPERATURES``,
``HENS_METHOD_SEQUENCE``, ``HENS_SYNTHESIS_QUALITY_TIER``, solver names,
tolerance, output formats, and run id. Do not pass persistent design-space or
solver controls as a separate runtime object to the design call.

Use ``enhanced_synthesis_method(quality_tier=...)`` for the recommended
quality-tier workflow, ``open_hens_method()`` for the original tier 1 OpenHENS
sequence, and ``network_evolution_method(initial_networks=...)`` for retrofit
evolution from an existing network.

When Couenne is unavailable for Couenne-backed stages, OpenPinch warns and
attempts the configured network-evolution route where possible.

Each exchanger stores shared design topology, area, and capital values once;
ordered parent-owned period-state records hold operating values. A
single-period network allows ``exchanger.state()`` and omitted query periods.
A multiperiod network requires the explicit identity for duty, temperature,
diagram, export, and controllability access.

Segmented Variable-Heat-Capacity Streams
----------------------------------------

A variable-heat-capacity process stream remains one physical parent on the
hot or cold solver axis. Its ordered internal segment records define the
local temperature--duty relation, heat-transfer coefficients, and exchanger
area calculations. Segment count therefore does not inflate physical stream,
match, exchanger, or stage-position counts.

HEN preparation carries ordered segment temperatures, cumulative duties,
local heat-capacity flowrates, heat-transfer coefficients, and deterministic segment
identities alongside the parent axes. Stage balances advance a cumulative
parent heat coordinate and map that coordinate through the piecewise
``T(Q)`` profile. Pinch decomposition may clip or split the active profile,
but it preserves the parent identity.

The current internal solver behavior is explicit:

- APOPT and Couenne use interval-disjunctive piecewise mappings.
- IPOPT uses active-segment refinement and repeats the continuous solve until
  the selected intervals stabilize.
- An unresolved active-segment solve is rejected with guidance to use APOPT or
  Couenne. OpenPinch does not silently substitute an average parent ``CP``.

The selected hot and cold utility parents may also be segmented. Their local
prices are integrated over the utility duty actually traversed, including a
partial final segment. Cost objectives, solved totals, verification, and
ranking therefore use the exact piecewise utility cost instead of multiplying
total duty by an average parent price. APOPT and Couenne use the interval
mapping; IPOPT uses the same active-segment refinement policy described above.
The current one-hot-utility and one-cold-utility HEN selection behavior is
unchanged; optimization across multiple utility parents is a future extension.

Segmented utility HTCs and temperatures are also retained in the post-solve
duty-aligned area slices. Flat utilities continue to behave as one virtual
segment and retain their previous objective and reporting results.

Each selected parent-level exchanger can expose ordered
``segment_area_contributions``. A contribution records its period, hot and
cold segment identities, slice duty, local endpoint temperatures, local heat-
transfer coefficients, LMTD, and area. The exchanger duty is the sum of its
local slices for the applicable period. Its multiperiod design area is the
maximum period-total slice area, not a sum of per-segment maxima drawn from
different periods.

.. code-block:: python

   for exchanger in design.network.exchangers:
       if exchanger.has_segment_area_contributions:
           print(exchanger.exchanger_id)
           print(exchanger.segment_area_by_period)
           print(exchanger.segment_design_area)

Grid diagrams and controllability remain parent-based. Segment details are
diagnostic metadata and do not become additional topology nodes.

Area Objective and Reported Area
--------------------------------

The nonlinear topology and total-cost objective retains the smooth Chen area
surrogate. After a solution is found, OpenPinch calculates the reported
exchanger area from ordered, duty-aligned segment slices using their local
terminal temperatures and heat-transfer coefficients. Those segment-summed
areas are also used for result verification and downstream ranking and
derivative calculations.

This separation is intentional: the Chen expression remains the accepted
optimization baseline, while exact local LMTD areas remain authoritative for
reported segmented exchangers. A possible future exact logarithmic-LMTD
formulation is limited to the continuous NLP path and is not part of the
current internal behavior.

Migration and Support Notes
---------------------------

Old import paths and OpenHENS field aliases have been removed from the runtime
API. OpenPinch does not provide runtime import aliases, OpenHENS field aliases,
or command parity with the original OpenHENS scripts. Use the conversion
scripts and converted JSON case inputs instead.
The old import paths should be treated as migration-only references, not as
compatibility aliases.

For development checks, use:

.. code-block:: bash

   pytest -m "not synthesis and not solver"
   pytest -m synthesis
   pytest -m solver

Next Steps
----------

- :doc:`../examples/notebook-series` for notebook 09.
- :doc:`../api/schemas-and-config` for ``TargetOutput.design`` schemas.
- :doc:`../api/service-layer` for the internal method-oriented service stack.
