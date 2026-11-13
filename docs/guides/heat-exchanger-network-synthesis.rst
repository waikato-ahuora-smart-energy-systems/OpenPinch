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
       dt_cont_multipliers=[1.0, 1.4, 1.8],
       stages=[3],
       best_solutions=3,
   )

   top = design.top(3)
   network = design.network(rank=1)
   grid = design.grid(rank=1)

The design view also exposes ``selected_network``, total recovery, hot and cold
utility duty, and ``utility(name)``. Serialize the complete result with
``design.result.model_dump(mode="json")``.

Exchanger Approach Temperatures
-------------------------------

Synthesis runs once per ``dt_cont_multipliers`` value
(``HENS_DT_CONT_MULTIPLIERS``; unset, the search width sets them: 0.75, 1.0
and 1.5 at the default width). A multiplier scales every
stream's, segment's and utility's ΔT contribution, as a zone ``dt_cont``
multiplier does in targeting, so 1.0 designs at the contributions the pinch
targets use, and a match between two streams needs the sum of their scaled
contributions. A stream whose ``dt_cont`` multiplier is locked keeps its own
contribution, as in targeting. A contribution of 0 runs at 0.1 K times the
multiplier. Results, manifests and networks record the multiplier; to reuse
an older approach-temperature grid, use multiplier = dTmin / (2 x
``THERMAL_DT_CONT``), so 14 K at the 5 K default is 1.4.

The multipliers set the pinch design step: the energy targets the seed
networks meet and the minimum approach of their exchangers. The thermal
derivative and network evolution steps that follow minimise total annual cost
and choose each exchanger's approach from the cost settings, down to 0.1 K.
With cheap area, exchangers can close well below the pinch design approach;
with dear area, they open up and the network uses more utility than the
target.

``HENS_MIN_EXCHANGER_APPROACH_SHARE`` raises that floor to a share of twice
the scaled ``THERMAL_DT_CONT``. It defaults to 0, which keeps the 0.1 K floor.
At 0.5, a multiplier of 1.0 with the 5 K default keeps every exchanger at 5 K
or more:

.. code-block:: python

   design = problem.design.heat_exchanger_network(
       dt_cont_multipliers=[1.0],
       options={"HENS_MIN_EXCHANGER_APPROACH_SHARE": 0.5},
   )

The share must be below 1: at 1, evolution cannot start from a pinch design
network whose exchangers sit exactly at that approach.
``verify_network_feasibility`` still checks approaches against the 0.1 K
floor, not against this share.

Several Hot or Cold Utilities
-----------------------------

Synthesis uses every hot and cold utility in the problem. Each heater and
cooler is one (stream, utility) pair at the end of the stream, with its own
duty, approach temperatures at both ends, price and area, so a stream can be
finished by the cheapest utility that reaches its target. In
``dairy_chilling_multiperiod.json`` the chilling streams, which end at 2 to
5 C, are cooled by glycol, while cooling water takes heat only where it can.

- A utility whose inlet cannot meet its approach at a stream's target in some
  period is never matched with that stream.
- Heaters and coolers on one stream run in parallel between the same stream
  temperatures. Synthesis does not place one utility after another along a
  stream, for example cooling water first and then glycol; each unit spans
  the stream's whole utility duty range.
- The pinch design method meets the utility targets and, where a side has a
  choice of utilities, prefers the cheaper ones.
- The default ``HU`` and ``CU`` that input preparation adds as backups take
  part only when they are the only utility on their side.

Named Advanced Methods
----------------------

.. code-block:: python

   wider = problem.design.heat_exchanger_network(width=2, depth=3)
   pinch_design = problem.design.pinch_design()
   thermal = problem.design.thermal_derivative(
       (pinch_design.selected_network,)
   )
   evolved = problem.design.network_evolution(
       (thermal.selected_network,)
   )

Called without seed networks, ``thermal_derivative()`` and
``network_evolution()`` start from the design already stored on the problem,
whichever method made it: its selected network first, then its other ranked
topologies, up to ``HENS_BEST_SOLUTIONS_TO_SAVE`` networks in all. So
``problem.design.pinch_design()`` followed by
``problem.design.network_evolution()`` evolves the pinch design networks.

Successful synthesis stores the serializable result on
``problem.results.design`` (the ``TargetOutput.design`` field). The design view
provides the selected network, ranked candidates, manifest, diagnostics, and
task metadata without requiring process engineers to call contributor services.

For multiple operating periods, call
``problem.design.multiperiod_heat_exchanger_network(...)`` after explicit
all-period targeting.

Heuristic Pinch Design
----------------------

The pinch design step, in ``problem.design.pinch_design()`` and the main
synthesis workflow, designs networks with the classic Linnhoff pinch design
rules by default. On each side of the pinch it:

- matches the streams at the pinch that cannot use a utility on that side
  (hot streams above the pinch, cold streams below), with partners whose CP
  is at least their own and no more such streams than partners, splitting
  streams where either rule fails;
- ticks each pinch match off at the smaller of its two loads, also allowing
  the match one stage further out when a split's mixed branches would break
  the approach;
- matches the remaining loads away from the pinch, the stream nearest the
  pinch first, with partners whose temperatures keep both ends of the match at
  the minimum approach (a pair matches again further out only when no new
  partner can take the load);
- gives the loads left on the other streams to utilities, allowing every
  utility that can reach them.

It enumerates the alternatives at each decision (pinch matches, optional
splits and partners away from the pinch), streams of similar CP first, up to
``HENS_HEURISTIC_PDM_NETWORKS`` networks per dt_cont multiplier (default
10). Each proposal is then solved by the pinch design side model with
only its matches allowed, so its duties and temperatures are consistent and
it meets the utility targets; the sides are joined, stage-reduced and
verified like the pinch design networks. Proposals the model cannot solve or
verify are dropped. With several operating periods, each period is designed
on its own and a proposal is the union of one design per period, one shared
topology.

In the main synthesis workflow every heuristic network seeds network
evolution directly (from search depth 1). Heuristic networks never go to the
thermal derivative method.

From search depth 2, or with ``HENS_PDM_MILP`` set, the MILP pinch design
method runs as well: it solves each side for minimum utility cost and then the
fewest units. From depth 2 its main networks go to the thermal derivative
method, and its extra minimum-unit networks (and from depth 3 all its
networks) straight to evolution; below depth 2, with ``HENS_PDM_MILP``, all
its networks seed evolution directly. With ``HENS_HEURISTIC_PDM_NETWORKS`` at 0 the MILP runs on its own.
At any depth, the MILP also runs at each dt_cont multiplier where the
heuristic realises no network, and its networks take the heuristic's place.
The heuristic's side models are solved with HiGHS when it is installed,
whatever ``HENS_SOLVER_PDM`` names.

.. code-block:: python

   heuristic_only = problem.design.pinch_design()
   with_milp = problem.design.pinch_design(options={"HENS_SEARCH_DEPTH": 2})

Search Width, Depth and Convergence
-----------------------------------

Synthesis searches in two directions, set by ``width`` and ``depth`` on
``heat_exchanger_network()`` or by ``HENS_SEARCH_WIDTH`` and
``HENS_SEARCH_DEPTH``. The width picks the dt_cont multipliers:

.. list-table::
   :header-rows: 1

   * - Width
     - dt_cont multipliers
   * - 0
     - 1.0
   * - 1 (default)
     - 0.75, 1.0, 1.5
   * - 2
     - 0.5, 0.75, 1.0, 1.5, 2.0
   * - 3
     - 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0

The depth picks the stages run at each multiplier and their effort:

.. list-table::
   :header-rows: 1

   * - Depth
     - Stages
     - Heuristic networks
     - Evolution breadth
   * - 0
     - heuristic pinch design only
     - 10
     - none
   * - 1 (default)
     - heuristic, then evolution
     - 10
     - 1
   * - 2
     - heuristic and MILP; MILP through the thermal derivative method
     - 10
     - 1
   * - 3
     - as 2; every MILP network also straight to evolution
     - 20
     - 2
   * - 4
     - as 3
     - 50
     - 3

The extra minimum-unit MILP networks are not part of the depth: set
``HENS_PDM_MIN_UNIT_NETWORKS`` and ``HENS_PDM_EXTRA_UNIT_NETWORKS`` (both 0 by
default) to add them at any depth that runs the MILP.

An option set explicitly overrides the level that would set it:
``HENS_DT_CONT_MULTIPLIERS`` (the ``dt_cont_multipliers`` argument) the width,
and ``HENS_HEURISTIC_PDM_NETWORKS``, ``HENS_PDM_MILP``,
``HENS_EVM_N_AD_BRANCHES`` and ``HENS_EVM_N_RM_BRANCHES`` the depth. A call's
``width`` and ``depth`` win over the call's ``options``, which win over the
stored options for that call only. A ``width`` or ``depth`` given in the call
also sets the options it governs, over stored ones, unless the same call sets
them: ``heat_exchanger_network(width=2)`` on a problem that stores
``HENS_DT_CONT_MULTIPLIERS`` runs the width-2 multipliers. The levels that ran
are in ``design.result.manifest.search_width`` and ``search_depth``. The other
``HENS_*`` keys, such as ``HENS_STAGE_SELECTION`` (the ``stages`` argument)
and ``HENS_SOLVE_TOLERANCE``, are listed with
their defaults in :doc:`../api/configuration-options`.

A solve that stopped before converging, for example at the IPOPT iteration
limit, is logged as an ``[Unconverged]`` warning. Its network still seeds the
steps that follow, but it is left out of the final ranking, so a low objective
from an unconverged model cannot win on cost. Networks that fail
``verify_network_feasibility`` are left out in the same way. When no converged,
feasible network remains, the design method raises ``WorkflowContractError``.

Duty Allocation on a Fixed Network
----------------------------------

When the network structure is already decided, define it by hand and let
OpenPinch allocate the duties. List recovery matches as
``(hot_stream, cold_stream, stage)`` with one-based stages, heaters by cold
stream and coolers by hot stream. A ``HeatExchangerNetwork``, such as a
previous design's ``selected_network``, is accepted in place of the mapping.
The example uses the four-stream ``problem`` loaded under Ranked Synthesis.

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

``optimise_duties`` keeps the matches and solves a non-isothermal stage-wise
model in which only duties and stream split fractions change.

Utility exchangers
   A heater or cooler entry is a stream name, ``(stream, utility)`` or
   ``(stream, utility, stage)``. Bare names use the ``hot_utility`` /
   ``cold_utility`` keys, which default to the problem's only hot and cold
   utility. A stream may have several utility exchangers, one per utility and
   position. Without a stage an exchanger sits at the stream end; with a stage
   it sits just after the stream leaves that stage, so ``("C2", "LPS", 2)``
   heats C2 between stages 2 and 1 and ``("H1", "CW", 1)`` cools H1 between
   stages 1 and 2. Exchangers at one position run in series: heaters from the
   coldest utility up, coolers from the warmest down. Each utility uses its own
   temperatures, heat-transfer coefficient and price.

``"utility"``
   Minimise total utility duty. Each exchanger keeps at least
   ``min_approach_temperature`` at both ends. Per-exchanger values in
   ``exchanger_approach_temperatures`` (keyed by ``exchanger_id``) override it.
   With neither, the stream temperature contributions set each limit.
``"area"``
   Minimise the total heat-transfer area with utility capped by
   ``max_hot_utility`` and/or ``max_cold_utility`` (kW, in every period). At
   least one cap is required. Each exchanger has one area shared by all
   operating periods: the largest area any period needs (exact LMTD). A period
   that needs less runs with a bypass, so its duties are achievable with that
   common area. Approach limits work as for ``"utility"``.
``"cost"``
   Minimise total annual cost ($/y): capital from the ``COSTING_HX_*``
   settings on the common areas, annualised with the capital recovery factor
   (``COSTING_DISCOUNT_RATE``, ``COSTING_SERVICE_LIFE``), plus utility price
   times duty over ``COSTING_ANNUAL_OP_TIME``, weighted over periods. All HEN
   synthesis methods cost networks this way. For benchmark data quoted in
   $/kW/y and $/y, set 1000 h/y, a zero discount rate and a one-year life.
   Only a positive
   approach is required at both ends of every exchanger;
   ``min_approach_temperature`` defaults to 1 K.

Zero-duty exchangers
   Every listed exchanger carries its approach constraint even at zero duty.
   When a solve leaves an exchanger at zero duty in every period, it is
   removed with that constraint and the problem is solved again, one
   exchanger at a time, until all remaining exchangers carry duty.
   ``selected_network.summary_metrics["removed_exchangers"]`` names them.

Segmented streams and utilities
   Streams with segments keep their piecewise temperature-heat profile: each
   segment has its own heat capacity, film coefficient and temperature
   contribution. The minimum approach is enforced at both ends of every
   exchanger and at every segment boundary inside it, and areas are summed
   over duty-aligned slices (``segment_area_contributions``). A segmented
   utility keeps its temperature profile and segment prices; its flow scales
   with use. Its profile shape comes from targeting, so it needs a targeted
   duty. Without a given approach, each side's largest segment contribution
   sets the limit. Segment kinks are rounded over 0.05 K in the solver.

Every process stream needs at least one exchanger. A utility that cannot meet
its approach against its stream at any duty is reported before solving. The
solver is ``HENS_SOLVER_EVM`` unless ``solver`` is passed. The network grid
draws each utility exchanger where it sits: at the stream end or on the
stage boundary it follows, with exchangers in series side by side.

Serialized Network Input
------------------------

The supported bridge carries the exact JSON-visible runtime dump through
``TargetInput.network``. Here ``network`` is the rank-1 network from Ranked
Synthesis:

.. code-block:: python

   import json

   from OpenPinch.contracts.input import TargetInput
   from OpenPinch.resources import read_sample_case

   case = json.loads(read_sample_case("Four-stream-Yee-and-Grossmann-1990-1.json"))
   network_payload = network.model_dump(mode="json")
   input_data = TargetInput.model_validate(
       {
           "streams": case["streams"],
           "utilities": case["utilities"],
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

Couenne and HiGHS use interval-disjunctive piecewise mappings. IPOPT uses
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
