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

The ranked networks are every network the run verified and costed, one per
topology (the cheapest): the networks each task ends with, and every network
an evolution search finds that improves on its best so far. Networks that
hold their utility levels at the targeted duties rank first, then the rest,
each group in ascending total annual cost; a total site network ranks by the
site's total annual cost (its net utility use after the site utility
balance).

Networks as They Are Found
--------------------------

Pass a ``HeatExchangerNetworkRecord`` as ``record=`` to see the networks while
synthesis runs. Each network enters the record as soon as it is verified and
costed in its final form: on a multiperiod median or period basis it is first
fitted to every period, and with an overdesign margin it is rated at the
margin. The record is thread-safe; another thread polls it:

.. code-block:: python

   import threading
   import time

   from OpenPinch.analysis.heat_exchanger_networks.results.record import (
       HeatExchangerNetworkRecord,
   )

   record = HeatExchangerNetworkRecord()

   def watch():
       version = 0
       while True:
           change = record.changed_since(version)
           if change is not None:
               version = change["version"]
               for entry in change["entries"]:
                   print(entry["key"], entry["method"], entry["total_annual_cost"])
               if change["status"] != "running":
                   return
           time.sleep(0.5)

   watcher = threading.Thread(target=watch)
   watcher.start()
   design = problem.design.heat_exchanger_network(record=record)
   watcher.join()

``record.version`` rises with every change. ``record.snapshot()`` returns the
whole record as new JSON-ready data: ``version``, ``status`` (``"pending"``,
``"running"``, ``"complete"`` or ``"failed"``), ``selected`` (the selected
entry's key once complete) and ``entries``, cheapest first, each with its
``key``, ``rank``, ``method``, ``task_id``, ``dt_cont_multiplier``,
``total_annual_cost``, ``unit_count``, ``version`` and ``network`` (the
network's JSON). ``record.changed_since(version)`` returns ``None`` when
nothing has changed, else the same header, only the entries added or replaced
since ``version``, and ``order``, every key in rank order. Entries are never
removed: a topology found again at a lower cost replaces its entry under the
same key, so a reader merges entries by key. Equal costs are ordered by task
id and then topology, so the contents and order do not depend on the order in
which worker processes finish.

On completion the result ranks exactly the record's networks and selects its
first. A record collects one call. With ``HENS_MAX_PARALLEL`` above 1, worker
processes send their networks to the calling process, which holds the record;
fitting and rating run there. Fitting and rating every improvement adds
solves, so a fitted or overdesigned design takes longer than one that fits
and rates the selected network alone did.

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
- Heaters and coolers at a stream's end run in parallel between the same
  stream temperatures, each with its share of the duty (reported as a split
  of the stream). With several hot or cold utilities, synthesis can also
  place a heater or cooler between two stages, so one utility follows
  another along a stream, and a sliding utility such as hot water is cut
  into slices that run in series.
- The pinch design method solves each side of the pinch at its utility
  target. With ``HENS_UTILITY_PLACEMENT: "price"`` a side with a choice of
  utilities prefers the cheaper ones; with ``"temperature"`` each utility
  level carries the duty targeting gives it. Heuristic networks and the extra MILP networks keep those targets.
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
15 rating solves. ``selected_network.overdesign`` reports the margin
(``percentage``), each period's recovery and utility before and after
(``periods``), the cost with and without the margin (``cost``,
``cost_without_margin``), every margin rated (``search``), for plotting
cost against margin, and every margin whose rating failed
(``failed_margins``). If no positive margin can be rated, ``"optimise"``
keeps 0 % and warns. ``design.optimise_duties(network, objective="rating")``
rates any network with installed areas the same way. A rating calibrates
each unit's capacity with Chen's LMTD at its design point, so off that point
it can credit a few per cent more duty than exact sizing allows; the rated
network is verified with that allowance, and
``summary_metrics["rating_area_allowance"]`` records the largest.

.. code-block:: python

   design = problem.design.heat_exchanger_network(overdesign_percentage="optimise")
   design.selected_network.overdesign.percentage

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
50). At search depths 0 and 1, when the option is unset, a case of more than
30 process streams gets fewer: 50 times (30 / streams) squared, at least 10
(10 for a 64-stream case). Each proposal is then solved by the pinch design
side model with only its matches allowed, so its duties and temperatures are
consistent and it meets the utility targets; the sides are joined,
stage-reduced and verified like the pinch design networks. Proposals the
model cannot solve or verify are dropped. A heuristic task has a wall-clock
limit of 300 s: past it, each side stops at its first realised proposal and
the joining at its first network, and a warning says the task kept the
networks found so far. With several operating periods, each period is designed
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
     - 2
   * - 2
     - heuristic and MILP; MILP through the thermal derivative method
     - 50
     - 3
     - 2
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

With more than one hot or cold utility, ``HENS_UTILITY_PLACEMENT`` sets how
synthesis shares duty between the levels. ``"temperature"`` places utility
as targeting does, as a site design needs: the least valuable (lowest
exergy) level serves first, and the pinch design holds each level at the
duty targeting gives it. ``"price"`` lets the pinch design choose the levels
by utility cost, as an isolated process would. ``"auto"`` (the default) is
``"price"`` when the design covers one process zone, for example
``heat_exchanger_network(zone="Hospital[S]")`` or a problem of one process,
and ``"temperature"`` when it covers more than one. A utility whose temperature slides,
such as hot water from 90 to 70 C, is then, by temperature, run through slices in series, so
one utility flow heats (or cools) several process streams in turn and
reaches streams a single heater on its whole range could not.
``HENS_UTILITY_SLICES: false`` turns the slices off: each heater and cooler
then runs its utility over the whole range, and a level it cannot hold is
chosen by price instead, with a warning.

``design.heat_exchanger_network()`` on a site pools every process's streams
as one process, so any two streams may match, and it warns when the problem
has more than one process zone. For a site of several
process zones, ``design.total_site_heat_exchanger_network(width=, depth=,
options=)`` designs each process alone with the pinch design method twice,
with its utilities chosen by price and held at the duties targeting gives
each level. At each dt_cont multiplier it joins one network per process from
each run into a lowest-cost and a level-held site seed, and refines the
seeds with network evolution as one problem, with no thermal derivative
step. Every process keeps each utility level at its seed's duty, heat is
recovered only between streams of one process, and the site pays for its
net utility use after the site utility balance, so the selected network may
follow price where that is cheaper for the site. ``depth`` sets the evolution's breadth, and from 2
the MILP pinch design method joins the heuristic for the process networks.
On a multiperiod site each process is designed over every period at once
(design basis ``"all"``, whatever ``HENS_DESIGN_BASIS`` says), so each level
carries its targeted duty in every period; the evolution holds those duties
period by period, with one area per exchanger sized for the period that
needs most, and no fit to the periods follows.

A site network carries the site utility balance. On each utility level, use
plus heat let down to lower levels plus heat dumped equals generation plus
heat let down from above plus imports, period by period. Imports enter hot
levels and mains (a hot and a cold utility of one name) at the import price,
within the hot utility's ``maximum_heat_flow`` when it has one; dumps leave
cold levels and mains at the dump price. So steam a process raises on a main
offsets steam another process uses there. A cold utility without segments
may have a negative price, a credit for the steam raised on it; a level with
a credit dumps no more than the processes raise on it, so the site cannot
buy steam to sell it. The selected network's ``summary_metrics`` hold ``site_capital_cost``, ``site_operating_cost`` and
``site_total_annual_cost``, and its ``source_metadata["site_utility_balance"]``
holds each level's use, generation, imports, letdowns and dumps per period.
Heat let down from a level to a colder one costs ``HENS_SITE_LETDOWN_PRICE``
($/MWh, annualised as utility prices are; default 0, so letdowns are free).
Network evolution of a site network holds each process's utility levels at
the duties its pinch design targeted. ``HENS_SITE_LEVEL_HOLDS: "at_least"``
holds each level at or above its target instead, so a process may take more
of a level, or raise more steam on one, when the site utility balance makes
that cheaper; the default, ``"fixed"``, holds them at their targets.
To design one process zone of a site alone, pass its name:
``design.heat_exchanger_network(zone="Food Plant[S]")``.

The site design uses the case's own utilities by default (``levels="given"``,
``HENS_SITE_LEVELS``). With ``levels="placed"`` it places the site's utility
levels by utility placement instead, once for each count in
``level_counts`` (``HENS_SITE_LEVEL_COUNTS``, default ``[2, 3, 4, 5]``), with
``placement`` passed on as the placement options. Each placed level serves
heating and cooling, as a steam main does. The case's sensible utilities (a
utility whose temperature changes by more than 1 K, such as hot water) stay
sensible while two isothermal levels remain, so on ``zonal_site``, with hot
water as its one sensible utility, count 2 places two isothermal levels and
count 4 three isothermal and one sensible. The site is designed with each
count's levels, and the cheapest design over the counts is selected (its site total annual cost, from site utility
accounting). A count whose
placement is infeasible, or whose design fails, is recorded and skipped.
The result ranks every network of each designed count by site cost, each tagged
``summary_metrics["utility_level_count"]``, and
``result.provenance.effective_settings["HENS_SITE_LEVEL_SWEEP"]`` lists every
count with its status (``designed``, ``infeasible``, ``failed`` or ``not
run``), its split and its cost. ``HENS_SITE_LEVEL_EARLY_STOP: true`` ends the
sweep once two designed counts in a row cost more than the cheapest before
them.

A placed level is a main: its hot and cold utility share one name
(``iso_1``, ``sensible_1``), so site utility accounting credits steam raised
on it against its use. The cheapest count's design is then refined by
level-temperature moves. A move shifts one placed level by
``HENS_SITE_LEVEL_STEP`` (default 5 K) up or down; it is refused if the
level would touch or pass another placed level or leave the range of the
site's stream temperatures. Each process's level duties are re-targeted at
the shifted level, and network evolution designs the site again from the
current best network's recovery matches, with the call's width, depth and
options. A round tries every placed level both ways and adopts the cheapest
move that lowers the site total annual cost; a move whose design fails is
recorded and skipped. Each set of level positions is designed once: a move
back to positions already designed, or whose design failed, is not designed
again. When no move improves, the step halves, and the moves
stop once it falls below ``HENS_SITE_LEVEL_STEP_MIN`` (default 1 K).
``HENS_SITE_LEVEL_STEP: 0`` moves no level. Every move tried, with its cost
and whether it was adopted, rejected, failed, refused (with the reason) or
skipped as already designed, is listed in
``result.provenance.effective_settings["HENS_SITE_LEVEL_MOVES"]``. The level
count stays fixed while levels move. Each move runs a site evolution, so the
moves can take several times as long as the sweep's design of one count.
``HENS_SITE_TIME_LIMIT`` (seconds, default none) bounds the sweep and the
moves together: once it runs out, no further count or move is started and
the cheapest design so far is returned; a design already running finishes,
and the first count is always designed. A ``levels="given"`` design is a
single design with no point to stop between, so the limit does not apply to
it; it warns when the limit, ``level_counts`` or ``placement`` is set, as it
ignores all three.

.. code-block:: python

    view = site.design.total_site_heat_exchanger_network(
        levels="placed",
        level_counts=[2, 3],
        placement={"evaluation_limit": 300},
        width=0,
        depth=1,
    )

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
   ``min_approach_temperature`` defaults to 1 K. The fit of a design to
   every period uses this objective, but keeps each exchanger at the case's
   minimum approach (its streams' temperature contributions); where that
   cannot be met it warns and fits at 1 K.
``"rating"``
   Minimise utility cost with every exchanger's area (heaters and coolers
   included) fixed at the network's installed ``area`` (each needs one). Every period's duty must
   fit that area; a period needing less runs with a bypass. The approach
   defaults to 1 K, as for ``"cost"``. Overdesign rates networks this way.

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
   for step, design in enumerate(result.frontier):
       print(
           step,
           design.modification_count,
           round(design.annual_savings),
           round(design.capital_cost),
           design.payback_years,
       )
   best = result.selected

The method follows network evolution, one structural change at a time,
with the network pinch approach of Smith's group deciding which changes to
solve:

1. The existing network is rated: every exchanger is held to its installed
   area and the duties minimise utility cost. This is the base case; its
   utility cost is the reference for savings.
2. Each step proposes, from every design kept so far, structural changes:
   new matches, joining a stage (which may split a stream) or in a new stage
   between two existing ones, and, with ``relocation_cost`` set, an existing
   recovery exchanger moved with its installed area to a match that shares
   one of its streams (its own pair at another position is a repiping). New
   matches are screened by the utility they could cut at first order: the
   heat they could recover where they would sit, but no more than their hot
   stream's cooling or their cold stream's heating, ties going to the larger
   heat potential.
3. The ``ceiling_candidates`` (default 12) screened changes are ranked by the
   utility cost their structure reaches with unlimited area (its network
   pinch), and the ``new_match_candidates`` (default 4) best are solved on
   the fixed-structure model with every existing exchanger free to grow: new
   exchangers are sized, added area is priced, and the objective is utility
   cost plus the annualised cost of the added area. An exchanger keeps added
   area of at least 1 % of its installed area (and 1 m2), each an area
   addition, and the structure is solved again with only those. Candidates
   are ranked by net annual benefit, savings less the annualised capital,
   and the best ``beam_width`` go on to the next step. Every solved design
   is checked as optimised duties are (approach temperatures, heat balances
   and areas, with installed areas widened by ``area_tolerance``) and
   dropped if it fails.

``result.frontier`` holds the best design found for each number of
structural changes, from the existing network (0) up to
``max_modifications``, so the trade-off between the size of the retrofit and
its benefit is visible; each design also lists the area additions it needs.
A design no better than one with fewer changes is marked ``dominated``.
``result.selected`` is the design with the best net annual benefit.

``strategy="stepwise"`` instead makes one modification per step, an area
addition on an exchanger whose duty its area limits being one, and solves
every screened change without a ceiling ranking. With Ipopt, whose duals
give each exchanger's area value (what one more m2 would save), it tries
added area only where that value is above zero, most valuable first; on the
validation cases this skips one to three solves and leaves the frontier
unchanged, as almost every exchanger at its area limit is worth enlarging.
It was the first version of the method and finds less on both validation
cases below.

Each design lists its ``modifications`` (``"new_match"``,
``"area_addition"``, ``"relocation"``, ``"enhancement"`` or ``"heat_pump"``,
the exchanger, the added area, fixed and area cost, any new splits, and for
a heat pump its ``heat_pump_cop``, ``heat_pump_capacity`` and
``heat_pump_refrigerant``), the rated network, hot and cold utility (kW), utility cost,
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
   ``beam_width`` (default 3) designs are kept per step. From each, a step
   solves the ``ceiling_candidates`` ceilings and up to two NLPs for each of
   the ``new_match_candidates`` best, about
   ``beam_width * (ceiling_candidates + 2 * new_match_candidates)`` small
   NLPs (stepwise: ``beam_width * (existing exchangers +
   new_match_candidates)``). ``min_approach_temperature`` (default
   1 K) applies to every exchanger.
Relocation
   ``relocation_cost`` (default ``None``: no relocations) is the fixed cost
   of moving an existing recovery exchanger, plus ``split_cost`` per stream
   the move splits; the moved exchanger keeps its installed area at no area
   cost. The default search ranks up to ``ceiling_relocations`` (default
   400) moves per design by their ceiling, where relocations and new
   matches each get ``new_match_candidates`` full solves. A simple score, the heat the new position could recover less the
   duty the exchanger gives up, barely tracks the ceiling: on Akpomiemie and
   Smith's case the paper's resequencing of exchanger 7 has the best ceiling
   of 395 moves but scores 127th. Ranking them all, the search finds that
   move and builds on it ($740,000/y net at two changes, against $712,000/y
   when only the 100 best-scored moves were ranked), at a cost: about 2,700
   solves instead of 889. ``HENS_MAX_PARALLEL`` (in ``options``) runs each
   step's solves, ceilings and sizing, on that many threads with the same
   result: with 8 the search took 37 s instead of 89 s serially. A large
   network offers far more moves: the 72-exchanger pulp mill offers 14,617
   from its existing network, so the cap, applied in score order, bounds
   the ranking; there three structural changes with relocation take 2,585
   solves and 164 s on 8 threads. The
   stepwise search tries the
   ``relocation_candidates`` (default 8) best-scored moves, and only those
   that gain more than they give up. On the Ciric and Floudas case
   relocation leaves the frontier unchanged, since most moves strand a
   stream whose utility exchanger is already at its area.
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
Heat transfer enhancement
   ``enhancement_factors`` maps an existing exchanger's id to the factor
   its heat transfer coefficient can be raised by, tube inserts for
   example, as limited by its geometry. An enhanced exchanger works as one
   with its area times the factor. Each costs ``enhancement_cost`` plus
   ``enhancement_area_cost`` per m2 of its installed area, and adds no area.
   The stepwise search tries enhancing each exchanger at its area limit as
   a step; the default search, after sizing a structure, sizes it again
   with the exchangers it would enlarge enhanced where they have a factor,
   and keeps the better design. Inserts raise pressure drop:
   ``enhancement_pressure_drops`` gives each exchanger's insert as the
   stream it is on and the pressure drop it adds (kPa), and
   ``pressure_drop_allowances`` the extra pressure drop each stream can take;
   no design's inserts exceed a stream's allowance. Hydraulics are not
   modelled beyond these sums: take the figures from a rating of each
   exchanger.
Repiping
   ``repiping_cost`` ($ per stream, default 0) prices the piping a step
   changes, which the unit costs leave out (Nguyen et al., 2010; Bütün et
   al., 2018 price it the same way): both streams of a new match, the
   streams a relocated exchanger leaves and joins (a stream it stays on in
   the same stage keeps its piping), and the stream of a new heater or
   cooler. Each modification lists them in ``repiped_streams``. A relocated
   exchanger keeps its id, so it can also take added area, which makes a
   move to a match that needs a bigger unit a resized reassignment.
Heat pumps
   ``heat_pump_cycle`` lets a step add a heat pump between two streams, as
   Bütün et al. (2018) retrofit heat pumping with the network: an
   evaporator at the end of a hot stream, evaporating
   ``min_approach_temperature`` below its target, and a condenser at the
   end of a cold stream, condensing as far above its target. The HPR
   service's cycle models rate it. ``"carnot"`` takes the heating COP
   ``1 + HPR_ETA_II_CARNOT * T_evap / (T_cond - T_evap)``, prices the
   compressor and package at ``heat_pump_cost`` per kW of condenser
   capacity and the two exchangers by the area law. ``"vapour_compression"``
   solves a single stage on each of ``HPR_REFRIGERANTS`` within the HPR
   cycle limits, keeps the fluid with the best COP and prices the whole
   machine (compressor, motor, evaporator and condenser) with the HPR
   service's installed capital. The work costs ``COSTING_HPR_ELE_PRICE``.
   The model links the two duties by the COP; the search screens stream
   pairs by the utility the heat pump could relieve, and its condenser and
   evaporator duties do not count as hot and cold utility. On the Ciric
   and Floudas case at $30/MWh electricity a Carnot heat pump from H1 to C2
   (COP 1.92, 14.7 MW) is the best first step, at $2.06M/y net; at the
   case's $100/MWh, and with a single water stage (COP 1.62, $13.7M
   installed), new matches win.
Sites and heat recovery loops
   When the existing network's streams span several process zones (a
   site's network given as a mapping), new matches and relocations join
   only streams of one process, and a ``UserWarning`` says the site utility
   balance is not modelled: utilities are priced unit by unit. Processes are
   linked only through a heat recovery loop, which ``HENS_RETROFIT_LOOPS``
   (default on) lets a step add: an intermediate fluid, hot water for
   example, at one temperature midway between a hot stream's target in one
   process and a cold stream's in another. A loop cooler at the end of the
   hot stream heats it and a loop heater at the end of the cold stream takes
   the same heat back, in every period and with no loss, so the hot stream
   reaches its target ``min_approach_temperature`` above the loop and the
   cold stream its target as far below (the loop circulates over 1 K, so
   the two targets must be at least twice the approach and 1 K apart; a
   pair closer than that needs a heat pump instead). Pairs are screened by
   the utility the loop could relieve: the hot stream's cooling, capped by
   the cold stream's heating. The two exchangers are priced like new
   exchangers, plus ``heat_recovery_loop_cost`` ($ per loop, for its pump
   and piping run; default 0) and repiping for both streams; the loop's
   heat is not counted as utility. Each loop is one modification of kind
   ``"heat_recovery_loop"`` with its ``loop_temperature`` (degC) and
   ``loop_duty`` (kW).

Heat pump targets
   ``design.retrofit_heat_pumps(existing, cycle=..., topology=...,
   options=...)`` asks the HPR service how much a heat pump could save
   before any search. The *ideal* target serves the process's heating
   target from its cooling target, as a new design could. The *as-piped*
   target serves only the existing heaters from only the existing coolers,
   at their solved temperatures and duties, so it needs no process-side
   repiping: each evaporator goes in series with a cooler, each condenser
   with a heater. Each target reports its levels, work and COP, and its hot
   utility, cold utility and operating cost before and after (utility
   prices and ``COSTING_HPR_ELE_PRICE``; no heat pump capital). Their gap
   splits into ``network_gap``, what the existing network costs beyond the
   targets before any heat pump, and ``heat_pump_gap``, how much less the
   as-piped heat pump saves than the ideal one. On the Ciric and Floudas
   case with Carnot cycles at the case's prices, the network gap is
   $2.84M/y. At full load the as-piped heat pump saves $0.31M/y against the
   ideal's $1.00M/y (COP 2.30 against 2.85, lifting to 271 degC for the
   existing heater); at half load it saves $0.95M/y against $1.08M/y. Most
   of the heat pump's value is reachable on the existing pipework, and the
   network's own recovery is worth more. Each target runs the HPR search,
   about 80 s each on this case.

Not covered: moving or repiping an existing heater or cooler, and new
utility exchangers placed between stages. Next to a paired new utility
exchanger, moving an existing one would mainly save the new unit's cost,
while its steam or cooling-water service usually makes it costlier to move
than a recovery exchanger. Area reduction is not a move: an exchanger may
always run below its area, and existing area is sunk.

The search is a greedy beam, so more candidates or ``new_utility_units`` do
not always end better. With the stepwise search on the Ciric and Floudas
case (report costs, four modifications), four new-match candidates reach
about $1.83M/y, eight end at $1.17M/y after a better first step, and all 24
with new utility exchangers reach $1.97M/y (496 solves). Compare a few
settings on a real case.

Validation
~~~~~~~~~~

Three published cases check the method; the tests run all three.

Ciric and Floudas (1989), as Example 1 of Hallberg and Scribner (2009)
   The existing network rates at 17,519 kW of hot utility against the
   published 17,759 kW. Under the report's rules (10 K approach, at most 20 %
   more area on an existing exchanger, $17,300 per added shell, at most two
   new exchangers) the best design with two new exchangers adds H1-C2 and
   H2-C2 matches and 20 % more area on E1 and E3: $1.91M capital and 5,532 m2
   of network area, against the report's MILP at $1.73M and 5,088 m2 with
   different matches. The report's savings figures cannot be compared: its
   NPV, annualised savings and return on investment do not reconcile.
Hallberg and Scribner (2009), Example 2, crude unit
   Thirteen process streams, eight of them with several segments, and 18
   existing exchangers. Its existing network rates at 68,666 kW of hot and
   74,835 kW of cold utility against the published 67,964 and 75,051 kW,
   once every installed area is widened by 5 % (``area_tolerance=0.05``):
   some published areas, E6's most, are a little short of their published
   duties. The report gives no retrofitted utility for this example, so
   only the existing network is checked.
Akpomiemie and Smith (2018), simplified crude preheat train
   Stream data, the existing network and its duties are from the paper
   (Tables 3 and 4, Figs. 10 and 19). Every stream has h = 0.5 kW/m2K and
   installed areas are back-calculated from the published duties, so U*A
   matches the paper. With unlimited area at a 10 K approach the existing
   structure saves 13.9 % of hot utility and the paper's retrofitted
   structure reaches 10,960 kW (24.2 %, every cooler at zero), both as
   published. The search with its defaults reaches the same hot utility in
   three structural changes (three new matches, with added area on four
   exchangers) in 141 solves, at $521k of capital; the paper does it with
   two (moving one exchanger and adding one) and more area on several. The
   paper's $4,000 fixed cost per unit makes new exchangers cheap. The
   stepwise search, counting each enlarged exchanger as a step, needs seven
   steps and every new-match position (811 solves), and stops at 20.5 %
   after six with four candidates.

   Enhancement (Section 5.1, Tables 5, 6 and 8): with the paper's factors
   for E5 (U from 0.325 to 0.448) and E1 (0.517 to 0.669), inserts at $500
   plus $10/m2 and no other changes allowed, the stepwise search enhances E5
   and then E1, as the paper does, and reaches 13,545 kW against its
   13,661.52 kW (5.7 % saved from the rated base against 5.5 %). Given E2's
   factor too and the paper's pressure drops (E5's insert adds 21.94 kPa to
   C1, E1's 1.96 and E2's 31.91, with C1 at 668.51 of its 700 kPa), the
   search never enhances E2, as the paper concludes; without the allowance
   it adds E2 as a third step. Combining enhancement with structural changes (Section 5.3),
   with relocation at $10,000 and factors for every exchanger the paper
   enhances: the first change becomes the paper's relocation of E7 with
   five exchangers enhanced ($762k/y net against $621k/y without
   enhancement), and at two changes enhancement cuts capital from $506k to
   $346k and added area from 1,473 to 932 m2, raising the net benefit from
   $740k/y to $934k/y. The paper reports about 65 % more retrofit profit
   from the combination, under its own sizing and pressure drop limits.

The default (structure-first) and stepwise searches compared (report costs; Smith's
Table 5 costs at a 10 K approach):

.. list-table::
   :header-rows: 1

   * - Case
     - Stepwise, 4 steps
     - Structure, 3 structural changes
   * - Ciric and Floudas, report costs
     - $1.83M/y net, 67 solves
     - $1.98M/y net, 133 solves
   * - Ciric and Floudas, report rules
     - $1.15M/y net, 66 solves
     - $1.29M/y net, 136 solves
   * - Akpomiemie and Smith
     - 11,777 kW hot utility, 76 solves
     - 10,960 kW (the published target), 141 solves

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
