Heat Exchanger Network Synthesis
================================

Install the HEN synthesis extra and the IDAES solver extensions before running
solver-backed methods:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"
   idaes get-extensions

Ranked Synthesis
----------------

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem(
       "Four-stream-Yee-and-Grossmann-1990-1.json",
       project_name="Four Stream",
   )
   design = problem.design.heat_exchanger_network(
       approach_temperatures=[10.0, 14.0, 18.0],
       stages=[3],
       best_solutions=3,
   )

   top = design.top(3)
   network = design.network(rank=1)
   grid = design.grid(rank=1)

The design view also exposes ``selected_network``, total recovery, hot and cold
utility duty, and ``utility(name)``. Serialize the complete result with
``design.result.model_dump(mode="json")``.

Named Advanced Methods
----------------------

.. code-block:: python

   enhanced = problem.design.enhanced_heat_exchanger_network(quality_tier=2)
   open_hens = problem.design.open_hens()
   pinch_design = problem.design.pinch_design()
   thermal = problem.design.thermal_derivative(
       (pinch_design.selected_network,)
   )
   evolved = problem.design.network_evolution(
       (thermal.selected_network,)
   )

Successful synthesis stores the serializable result on
``problem.results.design`` (the ``TargetOutput.design`` field). The design view
provides the selected network, ranked candidates, manifest, diagnostics, and
task metadata without requiring process engineers to call contributor services.

For multiple operating periods, call
``problem.design.multiperiod_heat_exchanger_network(...)`` after explicit
all-period targeting.

Duty Allocation on a Fixed Network
----------------------------------

When the network structure is already decided, define it by hand and let
OpenPinch allocate the duties. List recovery matches as
``(hot_stream, cold_stream, stage)`` with one-based stages, heaters by cold
stream and coolers by hot stream. Utility names default to the problem's only
hot and cold utility; add ``hot_utility``, ``cold_utility`` or ``stage_count``
keys to override them. A ``HeatExchangerNetwork``, such as a previous design's
``selected_network``, is accepted in place of the mapping.

.. code-block:: python

   structure = {
       "recovery": [
           ("Raw Milk", "Milk Concentrate", 1),
           ("HT Flash", "Milk Concentrate", 1),
           ("Raw Milk", "CIP Water", 2),
           ("HT Flash", "CIP Water", 2),
       ],
       "heaters": ["Milk Concentrate"],
       "coolers": ["Raw Milk", "HT Flash"],
   }

   min_utility = problem.design.optimise_duties(
       structure, objective="utility", min_approach_temperature=10.0
   )
   min_area = problem.design.optimise_duties(
       structure,
       objective="area",
       max_hot_utility=1.5 * min_utility.total_hot_utility,
   )
   min_cost = problem.design.optimise_duties(structure, objective="cost")

``optimise_duties`` keeps every listed exchanger and solves the stage-wise
model with fixed matches and free stream splits, so only duties and split
fractions change. Exchangers that end at zero duty are still reported, marked
inactive.

``"utility"``
   Minimise total utility use. Each exchanger keeps at least
   ``min_approach_temperature`` at both ends. Per-exchanger values in
   ``exchanger_approach_temperatures`` (keyed by ``exchanger_id``) override it.
   With neither, the stream temperature contributions set each match's limit.
``"area"``
   Minimise total heat-transfer area with total utility capped by
   ``max_hot_utility`` and/or ``max_cold_utility`` (kW). At least one cap is
   required. Approach limits work as for ``"utility"``.
``"cost"``
   Minimise total annual cost: annualised exchanger capital from the
   ``COSTING_HX_*`` settings plus utility cost. Only a positive approach is
   required at both ends of every exchanger; ``min_approach_temperature``
   defaults to 1 K.

The stage-wise model allows one heater per cold stream and one cooler per hot
stream, and every process stream needs at least one exchanger. Stream names
may be given without their zone prefix when unambiguous. The ``"area"``
objective optimises one operating period; pass ``period_id`` for a
multiperiod problem. The solver is ``HENS_SOLVER_EVM`` unless ``solver`` is
passed.

Serialized Network Input
------------------------

The supported bridge carries the exact JSON-visible runtime dump through
``TargetInput.network``:

.. code-block:: python

   from OpenPinch.contracts.input import TargetInput

   network_payload = network.model_dump(mode="json")
   input_data = TargetInput.model_validate(
       {
           "streams": stream_payloads,
           "utilities": utility_payloads,
           "network": network_payload,
       }
   )

   restored = TargetInput.model_validate_json(input_data.model_dump_json())
   assert restored.model_dump(mode="json")["network"] == network_payload

The nested value is a transport schema, not a synthesis seed. Private solver
and source metadata are absent from the dump and rejected if manually added.
Endpoint classifications use title-case ``StreamID`` values: ``Process`` and
``Utility``. ``Unassigned`` and legacy lowercase values are invalid.

Segmented Variable-Heat-Capacity Streams
----------------------------------------

A variable-heat-capacity process stream remains one physical parent on the hot
or cold solver axis. Its ordered internal segments define the local
temperature--duty relation, heat-transfer coefficients, and exchanger area;
segment count does not inflate physical stream, match, exchanger, or stage
counts.

HEN preparation retains ordered segment temperatures, cumulative duties, local
heat-capacity flowrates, heat-transfer coefficients, and deterministic segment
identities. Stage balances advance a cumulative parent heat coordinate through
the piecewise ``T(Q)`` profile. Pinch decomposition can split the active profile
while preserving the one physical parent identity.

APOPT and Couenne use interval-disjunctive piecewise mappings. IPOPT uses
active-segment refinement and repeats the continuous solve until the selected
intervals stabilize. An unresolved active-segment solve is rejected with solver
guidance; OpenPinch does not silently substitute an average parent ``CP``.

Each selected parent-level exchanger can expose ordered
``segment_area_contributions``. A contribution records its period, hot and cold
segment identities, slice duty, local endpoint temperatures, local heat-transfer
coefficients, LMTD, and area. The multiperiod design area is the maximum
period-total slice area, not a sum of segment maxima taken from different
periods.

Area Objective and Reported Area
--------------------------------

The nonlinear topology and total-cost objective retains the smooth Chen area
surrogate. After solving, OpenPinch calculates reported exchanger area from
ordered duty-aligned slices with their local terminal temperatures and
heat-transfer coefficients. These segment-summed areas are used for result
verification, ranking, and derivative calculations. A future exact
logarithmic-LMTD formulation would be limited to the continuous NLP path and is
not the current contract.


Contributor verification separates ordinary, synthesis, and external-solver
profiles:

.. code-block:: bash

   pytest -m "not synthesis and not solver"
   pytest -m synthesis
   pytest -m solver

See notebooks 15 through 17 in :doc:`../examples/notebook-series`.
