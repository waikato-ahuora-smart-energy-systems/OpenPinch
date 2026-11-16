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
- The pinch design method solves each side of the pinch at its utility
  target and, where a side has a choice of utilities, prefers the cheaper
  ones. Heuristic networks and the extra MILP networks keep those targets.
  The main MILP network then drops stages that carry a small share of its
  recovery, which can move that recovery to utility (about 10% more utility
  on the ten-stream Ahmad case).
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
``problem.design.network_evolution()`` evolves the pinch design networks:
only the selected one at the default ``HENS_BEST_SOLUTIONS_TO_SAVE`` of 1.

Successful synthesis stores the serializable result on
``problem.results.design`` (the ``TargetOutput.design`` field). The design view
provides the selected network, ranked candidates, manifest, diagnostics, and
task metadata without requiring process engineers to call contributor services.

Multiple Operating Periods
--------------------------

On a case with several operating periods, ``heat_exchanger_network()``
designs one network that serves every period. ``design_basis`` (stored as
``HENS_DESIGN_BASIS``) sets what it is synthesised from:

- ``"median"`` (the default): a synthetic single-period case whose streams
  take their period-weighted medians, of supply and target temperature, heat
  flow, film coefficient and ``dt_cont``, over the operating periods (weight
  above zero, some stream running). With ``basis_clusters`` (stored as
  ``HENS_BASIS_CLUSTERS``) above 1, the operating periods are first grouped
  by weighted k-medoids on each stream's temperatures, heat flow and running
  flag, and the median is over the heaviest group; ``HENS_BASIS_CANDIDATES``
  ``"all"`` designs from every group's case and keeps the cheapest. A stream
  running in less than ``HENS_BASIS_ACTIVE_SHARE`` (0.5) of the group's
  weight is left out; utilities come from the group's medoid period.
- ``"period"``: one period alone, the ``period_id`` given, for example the
  peak or the most frequent period. Passing ``period_id`` implies
  ``"period"``.
- ``"all"``: every period at once, with no fit. Synthesis targets every
  period first when the problem has no period results.

A median or period design is then fitted to every period: each exchanger
gets one area, the largest any period needs, and a period needing less runs
with a bypass. A stream with no utility at its end first gets the cheapest
one that can serve it there in every period; one no period uses is removed.

.. code-block:: python

   basis = problem.design.design_basis(basis_clusters=2)  # preview, no synthesis
   problem.design.heat_exchanger_network()  # the median basis
   problem.design.heat_exchanger_network(period_id="peak")
   problem.design.heat_exchanger_network(design_basis="all")

``overdesign_percentage`` (stored as ``HENS_OVERDESIGN_PERCENTAGE``) adds a
margin to every heat-recovery area; heaters and coolers are not scaled. The
network is then rated over every period at the larger areas: each period's
duties are re-optimised for the least utility cost, with a bypass where a
period needs less. ``"optimise"`` chooses the margin, between 0 and
``HENS_OVERDESIGN_MAX`` (50 %), for the least total annual cost, from about
15 rating solves. ``selected_network.summary_metrics["overdesign"]`` reports
the margin, each period's recovery and utility before and after, the cost
with and without the margin, and every margin tried, for plotting cost
against margin. ``design.optimise_duties(network, objective="rating")`` rates
any network with installed areas the same way.

.. code-block:: python

   design = problem.design.heat_exchanger_network(overdesign_percentage="optimise")
   design.selected_network.summary_metrics["overdesign"]["percentage"]

The result's provenance records the basis (``HENS_DESIGN_BASIS``, with
``basis_period_id`` for a period basis, or ``HENS_BASIS_CLUSTERS``,
``basis_period_ids`` and ``basis_medoid`` for a median one) and lists every
period.

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

A side left with fewer complete proposals than the cap is searched again by
two fallbacks in turn, least intrusive first, and their new proposals follow
the first search's: the first also tries the next nearest must streams to the
pinch, not just the nearest; the second lets a stream that no partner can take
in series split a partner away from the pinch, matching it in parallel with
that partner's last match where the stage keeps its approach. A side the first
search fills is not searched again.

In the main synthesis workflow the cheapest heuristic networks seed network
evolution directly (from search depth 1): ``HENS_HEURISTIC_PDM_EVOLUTION_SEEDS``
per dt_cont multiplier, 3 at depths 1 and 2, set by the depth (see below).
The rest stay final candidates as they are. Heuristic networks never go to the
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
     - Heuristic networks evolved
     - Evolution breadth
   * - 0
     - heuristic pinch design only
     - 50
     - none
     - none
   * - 1 (default)
     - heuristic, then evolution
     - 50
     - 3
     - 1
   * - 2
     - heuristic and MILP; MILP through the thermal derivative method
     - 50
     - 3
     - 1
   * - 3
     - as 2; every MILP network also straight to evolution
     - 50
     - 20
     - 2
   * - 4
     - as 3
     - 50
     - 50
     - 3

The extra minimum-unit MILP networks are not part of the depth: set
``HENS_PDM_MIN_UNIT_NETWORKS`` and ``HENS_PDM_EXTRA_UNIT_NETWORKS`` (both 0 by
default) to add them at any depth that runs the MILP.

An option set explicitly overrides the level that would set it:
``HENS_DT_CONT_MULTIPLIERS`` (the ``dt_cont_multipliers`` argument) the width,
and ``HENS_HEURISTIC_PDM_NETWORKS``, ``HENS_HEURISTIC_PDM_EVOLUTION_SEEDS``,
``HENS_PDM_MILP``, ``HENS_EVM_N_AD_BRANCHES`` and ``HENS_EVM_N_RM_BRANCHES`` the
depth. A call's
``width`` and ``depth`` win over the call's ``options``, which win over the
stored options for that call only. A governed option stored on the problem
also wins over a stored level: a stored ``HENS_PDM_MILP: false`` keeps the
MILP off even with a stored depth of 2. A width or depth given in the call, as an
argument or in its ``options``, also sets the options it governs, over stored
ones, unless the same call sets them: ``heat_exchanger_network(width=2)`` on a problem that stores
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
feasible network remains, the design method raises ``WorkflowContractError``;
if a solve stopped at a limit, the error names the solver options that raise
it.

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

Retrofit of an Existing Network
-------------------------------

``design.retrofit`` starts from an existing network and finds cost-effective
modifications that recover more heat. Give the network as for
``optimise_duties``, with every exchanger's installed area in m2: entries may
be mappings with ``area`` and an optional ``id`` (a plant tag), or a
``HeatExchangerNetwork`` whose exchangers carry ``area``. The example is
Example 1 of Hallberg and Scribner (2009), adapted from Ciric and Floudas
(1989), packaged as ``Five-stream-Ciric-and-Floudas-1989.json``; notebook 16
runs it.

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("Five-stream-Ciric-and-Floudas-1989.json")
   existing = {
       "recovery": [
           {"hot": "H3", "cold": "C2", "stage": 1, "area": 609.7, "id": "E1"},
           {"hot": "H3", "cold": "C1", "stage": 2, "area": 579.2, "id": "E2"},
           {"hot": "H1", "cold": "C1", "stage": 1, "area": 1008.5, "id": "E3"},
           {"hot": "H2", "cold": "C2", "stage": 2, "area": 117.96, "id": "E4"},
       ],
       "heaters": [{"stream": "C2", "area": 246.75, "id": "E7"}],
       "coolers": [
           {"stream": "H1", "area": 787.5, "id": "E5"},
           {"stream": "H2", "area": 104.6, "id": "E6"},
       ],
   }

   result = problem.design.retrofit(
       existing,
       max_modifications=3,
       area_addition_cost=8_650.0,
       split_cost=10_000.0,
   )
   for design in result.frontier:
       print(
           design.modification_count,
           round(design.annual_savings),
           round(design.capital_cost),
           design.payback_years,
       )
   best = result.selected

The method follows network evolution, one modification at a time:

1. The existing network is rated: every exchanger is held to its installed
   area and the duties minimise utility cost. This is the base case; its
   utility cost is the reference for savings.
2. Each step tries, from every design kept so far, added area on each
   existing exchanger whose duty is limited by its area, and the most
   promising new matches, joining a stage (which may split a stream) or in a
   new stage between two existing ones. New matches are ranked by the
   utility they could cut at first order: the heat they could recover where
   they would sit, but no more than their hot stream's cooling or their cold
   stream's heating. Ties, including matches that cut nothing directly, go
   to the larger heat potential. With ``relocation_cost``
   set, a step may also move an existing recovery exchanger, with its
   installed area, to a screened match that shares one of its streams; its
   own pair at another position is a repiping.
3. Each candidate is solved on the fixed-structure model: existing exchangers
   keep their installed area unless they are given more, new exchangers are
   sized, and the objective is utility cost plus the annualised cost of the
   added area. Candidates are ranked by net annual benefit, savings less the
   annualised capital, and the best ``beam_width`` go on to the next step.
   Every solved design is checked as optimised duties are (approach
   temperatures, heat balances and areas, with installed areas widened by
   ``area_tolerance``) and dropped if it fails.

``result.frontier`` holds the best design found for each number of
modifications, from the existing network (0) up to ``max_modifications``, so
the trade-off between the size of the retrofit and its benefit is visible.
A design no better than one with fewer modifications is marked
``dominated``. ``result.selected`` is the design with the best net annual
benefit.

Each design lists its ``modifications`` (``"new_match"``,
``"area_addition"`` or ``"relocation"``, the exchanger, the added area, fixed and area cost, and
any new splits), the rated network, hot and cold utility (kW), utility cost,
annual savings, capital cost ($), annualised capital and net annual benefit
($/y), and the simple payback in years.

Costs
   Added area, on an existing or a new exchanger, costs
   ``COSTING_HX_AREA_COEFF * area ** COSTING_HX_AREA_EXP``; existing area is
   sunk. A new exchanger adds ``new_unit_cost`` (default
   ``COSTING_HX_UNIT_COST``) and ``split_cost`` per stream it splits; more
   area on an existing exchanger adds ``area_addition_cost``. Capital is
   annualised with the capital recovery factor of ``COSTING_DISCOUNT_RATE``
   and ``COSTING_SERVICE_LIFE``.
   ``max_added_area_fraction`` (default ``None``, no cap) limits the area
   added to an existing exchanger to that fraction of its installed area,
   for example 0.2 where a shell can take at most 20 % more tubes.
Area tolerance
   Plant areas are rounded and the solver uses Chen's LMTD approximation, so
   an existing network may fall just short of its own duties at its exact
   areas. ``area_tolerance`` (default 0.02) widens every installed area by
   that fraction. An existing network that still cannot meet its targets
   raises ``ValueError``.
Search size
   ``beam_width`` (default 3) designs are kept per step and
   ``new_match_candidates`` (default 4) new matches are tried from each, so a
   step solves up to ``beam_width * (existing exchangers +
   new_match_candidates)`` small NLPs. ``min_approach_temperature`` (default
   1 K) applies to every exchanger.
Relocation
   ``relocation_cost`` (default ``None``: no relocations) is the fixed cost
   of moving an existing recovery exchanger, plus ``split_cost`` per stream
   the move splits; the moved exchanger keeps its installed area at no area
   cost. ``relocation_candidates`` (default 8) moves are tried from each
   design, ranked by the heat the new position could recover less the duty
   the exchanger gives up. Relocation can enlarge the search several-fold:
   on the Ciric and Floudas case with 30 candidates it takes 319 solves
   instead of 48, and leaves the frontier unchanged, since most moves strand
   a stream whose utility exchanger is already at its area.
New utility exchangers
   A new heater or cooler never lowers utility cost by itself, but it can
   free the recovery units a move needs: a cooler on a hot stream that has
   none lets that stream's recovery exchangers give up duty. With
   ``new_utility_units=True`` each new match and relocation is also tried
   together with each new end heater (on a cold stream without one) or
   cooler (on a hot stream without one). The pair is one modification: one
   more ``new_unit_cost`` plus the utility exchanger's area, reported as
   ``utility_exchanger_id`` and ``utility_area``. A pair whose utility
   exchanger carries no duty is dropped. On the Ciric and Floudas case a
   cooler on H3 lets a new H1-C2 match in a new third stage save about
   $836k/y instead of $222k/y alone.

Not covered: moving or repiping an existing heater or cooler, and new
utility exchangers placed between stages. Next to a paired new utility
exchanger, moving an existing one would mainly save the new unit's cost,
while its steam or cooling-water service usually makes it costlier to move
than a recovery exchanger. Area reduction is not a move: an exchanger may
always run below its area, and existing area is sunk.

The search is a greedy beam, so more candidates or ``new_utility_units`` do
not always end better. On the Ciric and Floudas case with the report's costs
and four modifications, the defaults reach about $1.83M/y (67 solves);
eight new-match candidates end at $1.17M/y after a better first step; and
trying all 24 positions with new utility exchangers reaches $1.97M/y (496
solves). Compare a few settings on a real case.

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
