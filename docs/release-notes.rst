Pre-Release Notes
=================

Unreleased
----------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The vapour-compression warm starts (parallel, cascade and VC-MVR) now
  seed each evaporator ``dt_phase_change`` below the Carnot evaporating
  temperature. A Carnot evaporator at ``T`` spans
  ``[T - dt_phase_change, T]`` but a vapour evaporator widens upwards from
  its saturation temperature, so the seed sat 0.01 K warmer than the Carnot
  design. On a utility background, where a refrigeration band is 0.1 K
  wide, that lost 10% of the band's duty to refrigeration. On a brewery
  utility-placement case the parallel seed now costs 431.4 k$/y rather than
  446.6 k$/y (refrigeration 79.0 kW rather than 105.5 kW), and a search that
  kept the seed kept the dearer design.

Zones and targeting
~~~~~~~~~~~~~~~~~~~

- A process zone that holds a unit operation kept only the unit
  operation's streams: the streams placed on the process zone itself
  vanished from it and from the site, so their targets left them out. Both
  are now kept.
- Re-importing subzone streams no longer drops a zone's own cold stream
  whose key matches an imported hot stream's key (or the reverse): hot and
  cold imported keys are now tracked separately.
- Re-importing subzone streams keeps their keys and names when a zone's own
  stream and a deeper subzone's stream share a name. Before, the collision
  rename at the site leaked into the next import, so ``U.H1`` became
  ``U.H1_1``.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A problem with a process MVR component can now be designed
  (``design.pinch_design`` and ``design.heat_exchanger_network`` raised
  before, with the component active or deactivated). Two faults:
  ``problem.components.add_process_mvr`` keeps the vapour stream it
  replaces in the zone, inactive, beside the compressed stream, and the
  synthesis arrays took both, so the vapour's heat was counted twice; they
  now take active streams only, as targeting does. And a stream with a
  single segment, as the compressed stream is, got the zero flowrate
  placeholder of a segmented stream while the models treated it as
  unsegmented, so it carried no heat; it now keeps its own flowrate. On
  the packaged ``process_mvr.json`` with the component active, pinch design
  at a 0.75 multiplier recovers 120 kW and cools 30.2 kW. Its default
  utilities sit 5 K from the stream targets, so the 1.0 and 1.5
  multipliers still find no network.
- A retrofit now proposes matches on streams with several segments. Such a
  stream has no single heat capacity flowrate, and the search read it as
  0, so every match on it screened to nothing: on the Hallberg and
  Scribner crude case (eight of thirteen streams segmented) the crude feed
  was never matched. The screens now take a segmented stream's duty over
  its temperature span. On that case the first step is a new match on the
  crude feed that cuts hot utility from 68.9 to 51.9 MW, and two steps
  reach 46.0 MW. A candidate whose solved duty the model cannot place on a
  segment profile is dropped; before, the error it raised ended the
  search.
- A retrofit no longer depends on which operating period is listed first.
  The search read every stream's temperatures and heat capacity flowrate
  from the first period, so a stream that is off in that period had a
  flowrate of 0 and every match on it screened to nothing: a two-period
  case with a cold stream off 10 % of the time found no step with the idle
  period first and a new match worth 32 k$/y with it last. Stream
  flowrates are now the period-weighted mean and stream temperatures the
  weighted mean over the periods the stream runs. The new-match, heat pump
  and heat recovery loop screens and the relocation scoring read a solved
  network's duties and temperatures as period-weighted means too, so a
  heater that is off in the first period no longer hides the heat it could
  give up. The utility a match, heat pump or loop could relieve pairs the
  hot stream's cooling with the cold stream's heating in each period before
  the mean, so a link between streams that never run together relieves
  nothing. A new match's potential is likewise read in each period, from
  that period's flowrates and temperatures, and caps its relief period by
  period, so a pair of streams that never run together is not proposed.
  Relocation scoring still reads a match's potential from the period means.
- The default (structure) retrofit search now drops area additions that do
  not pay. The sizing solve weighs an addition's area cost against the
  utility it saves but not its fixed cost (``area_addition_cost``), so it
  grew units that did not earn that cost. Each step's best design is now
  solved again without each of its area additions in turn, and an addition
  whose removal raises the net annual benefit is dropped. On the Ciric and
  Floudas case at the report's costs the three-change design drops 112 m2
  on E1 and 33.5 m2 on E4: net annual benefit 1,960,807 to 1,963,253 $/y,
  capital 3.97 to 3.87 M$, for 11 more solves (134 to 145). The guide's
  solve counts for the structure search are updated.
- A retrofit result now reports ``failed_solves``: for each search step,
  the candidates dropped because their solve did not converge or their
  solved network failed its checks. They were dropped silently before.
  The search warns when more than a quarter of its solves failed: on the
  Ciric and Floudas case with relocation at the report's costs, 136 of 491
  did (28 %); without relocation 1 of 145.
- The minimum approach of a match is now the sum of its two sides'
  temperature contributions (each scaled by the dt_cont multiplier), with a
  floor of 0.1 K on the pair, for recovery matches, heaters and coolers.
  Before, a contribution of 0 was replaced by 0.1 K times the multiplier on
  the stream itself, so every match of a zero-contribution stream or
  utility ran that much wider than its partner's contribution: a 5 K
  stream on a 0 K utility at 5.1 K, now 5.0 K. Pinch design targets move
  with it, since the copy of the case they are computed on keeps zero
  contributions at zero too, except where a hot and a cold contribution sum
  below 0.1 K: each is then targeted at 0.05 K, and the streams split at the
  pinch with it, so the targets never ask for more recovery than the 0.1 K
  floor allows. On ``zonal_site.json``, whose six utilities
  have a contribution of 0, hot water now splits at 87 and 85 degC (87.1
  and 85.1 before). In the thermal derivative and evolution steps, which
  scale every contribution down to a nominal 0.1 K, a pair below the
  reference ``THERMAL_DT_CONT`` was held to less than the documented
  0.1 K (a 2.5 K stream on a 2.5 K stream to 0.05 K at the 5 K default);
  it is now held to 0.1 K. ``ZERO_CONTRIBUTION_REPLACEMENT`` is removed;
  ``pair_minimum_approach`` and ``MINIMUM_PAIR_APPROACH`` replace it.
- ``HENS_MIN_EXCHANGER_APPROACH_SHARE`` now defaults to 0.25 (was 0). The
  thermal derivative and network evolution steps hold each match to
  ``max((dtcont_src + dtcont_snk) x m x share, 0.1 K)``: a quarter of its
  pinch design approach by default, where before every match could close
  to 0.1 K. Evolved networks therefore keep wider approaches and use more
  utility and less area than before; set the option to 0 to let every
  match close to 0.1 K again. (At 0 a pair whose contributions sum above
  twice ``THERMAL_DT_CONT`` was held to more than 0.1 K, 0.2 K for two 10 K
  streams at the 5 K default; it is now 0.1 K like any other.) A fit to
  every period, a rating and an overdesign of an evolved network hold the
  same pair values (each pair's contributions at multiplier x share)
  instead of one uniform approach, which could sit above a pair's own
  value and force the fit down to its 1 K fallback.
  Each such network records the share it was designed with
  (``source_metadata["min_exchanger_approach_share"]``), so
  ``optimise_duties``, the fit and overdesign hold it whatever the
  configuration says later.
- Network evolution now adopts the areas of staged heaters and coolers with
  the rest of a better candidate. It kept the areas they had before, so a
  staged unit the candidate no longer used still counted in the area check
  and the evolved multiperiod network was rejected ("verification failed:
  area").
- The packaged ``Four-stream-Yee-and-Grossmann-1990-1.json`` sample case no
  longer sets ``HENS_MAX_PARALLEL`` to 10, so it runs with the default of
  one worker. On a case this small the worker processes cost more than they
  save: the default design took 11.3 s with 10 workers and takes 3.2 s with
  one, for the same two ranked networks. Set ``HENS_MAX_PARALLEL`` in the
  case's ``options`` to use worker processes.
- A synthesis run with ``HENS_MAX_PARALLEL`` above 1 now starts one pool of
  worker processes for the whole run, not one for each stage. Starting a
  pool takes seconds: with 10 workers the four-stream case started two
  pools at the default search depth and took 10.4 s, and five pools at
  depth 2 and took 24.7 s; it now starts one pool and takes 7.4 s and
  10.9 s, for the same networks. A pool a stage fails in is replaced at
  the next stage.
- A multiperiod design fits each network structure once at each design
  approach. A fit to every period sets duties and areas afresh, so networks
  that share a structure, the approach they were designed to and the levels
  they hold have the same fit, but each was fitted again. The packaged
  ``crude_preheat_train_multiperiod.json`` ran 26 fits for 5 ranked networks
  and now runs 21 (9.2 s to 7.9 s); ``dairy_chilling_multiperiod.json`` ran
  30 for 11 and now runs 17 (8.8 s to 6.8 s). The ranked networks and their
  costs are unchanged.
- A network fitted to every period, or rated at an overdesign margin, keeps
  the method, dt_cont multiplier and approach share it was designed with.
  Every such network reported ``network_evolution_method`` and no
  multiplier, because the fit and the rating run as network evolution
  tasks; a heuristic pinch design made at a multiplier of 1.5 and fitted on
  a multiperiod case read as an evolution network designed at 1.0. A later
  ``optimise_duties(objective="rating")`` of a fitted network therefore
  holds the approach the network was designed to.
- **Breaking:** ``HENS_PDM_STAGES`` now sets the MILP pinch design stage
  counts above and below the pinch (two values, which may be equal; unset
  lets each side take the count its streams need).
  ``HENS_STAGE_SELECTION``, the ``stages`` argument, no longer does: a list
  of exactly two values used to fix them as a side effect, could not hold
  two equal counts, and a longer or shorter list did nothing. A call that
  passed ``stages=[2, 3]`` for this now passes
  ``options={"HENS_PDM_STAGES": [2, 3]}``. ``HENS_STAGE_SELECTION`` keeps
  its one other use, the stage counts of the direct network evolution runs
  made when Couenne is missing.
- The thermal derivative step (search depth 2 and above) now minimises
  utility cost when a problem has more than one hot or more than one cold
  utility. It minimised hot utility duty, which is the same thing with one
  utility a side but leaves the split between two free. On the four-stream
  case with warm water added at a third of the cooling water price, its
  network at a multiplier of 1 put all 1,996 kW of cooling on cooling
  water (57.7 k$/y of utilities); it now cools on warm water (37.7 k$/y)
  at the same hot utility. Network evolution already corrected the split,
  so the selected design on that case is unchanged; the thermal derivative
  networks in the ranking are cheaper. Problems with one hot and one cold
  utility are unaffected.
- A split branch whose stream has a segment under twice the tolerance
  was floored above the whole stream's flow, so the branch carried more
  duty than its stream. The floor is now capped at the whole stream.
- Stage packing no longer empties every stage after one with no allowed
  match: packing chains only the stages that can carry duty.
- Verification now checks a network's capital cost against the sum of its
  exchangers' capital costs when every exchanger records one, so a network
  costed below its exchangers fails. Capital is not recomputed from the
  areas nor utility cost from the duties, as the network does not carry the
  cost law or the prices; the docstring no longer claims it catches every
  under-costed network.
- An infeasible fixed-structure solve is named by its objective in the
  error (as a failed one already was), not as ``network_evolution_method``.
- ``HENS_SITE_LETDOWN_PRICE`` ($/MWh, default 0) prices heat that a total
  site network's utility balance lets down from one level to a colder one.
  Letdowns were always free: the site balance had a letdown price, but no
  setting reached it. The default keeps today's results.
- The run manifest records ``pdm_stage_pair_limit`` only when the MILP
  pinch design stage ran at every multiplier, and ``tdm_parent_limit``
  only when the thermal derivative stage ran; otherwise both are ``None``.
  Before, a depth 1 run recorded both though neither stage ran.
- A repeated ``HENS_DT_CONT_MULTIPLIERS`` value no longer adds a second,
  identical direct network evolution task when Couenne is missing.
- ``design.retrofit`` on a multi-process site network given as a mapping
  (the documented form, which carries no site metadata, so the site guard
  let it through) proposed new matches between streams of different
  processes and priced utilities unit by unit with no warning. On a
  two-process site its first step matched the Cooling process's ``H3``
  with the Heating process's ``C1``. When the existing network's streams
  span several process zones, new matches and relocations are now screened
  only within one process, and a ``UserWarning`` says the site utility
  balance is not modelled. Single-process retrofits are unchanged.
- A retrofit whose existing network spans several process zones may now
  propose a new heat recovery loop, the only link it proposes between
  processes: an intermediate fluid at one temperature, midway between a hot
  stream's target in one process and a cold stream's in another (the
  targets at least twice ``min_approach_temperature`` and 1 K apart), heated
  by a loop cooler at the end of the hot stream and giving the same heat,
  in every period and with no loss, to a loop heater at the end of the cold
  stream. The two exchangers are priced like new exchangers, plus the new
  ``heat_recovery_loop_cost`` ($ per loop, its pump and piping, default 0)
  and repiping; the loop's heat is not counted as utility. The modification
  has kind ``"heat_recovery_loop"`` and reports ``loop_temperature`` (degC)
  and ``loop_duty`` (kW). ``HENS_RETROFIT_LOOPS`` (default on) turns the
  proposals off. On a two-process case whose only option is a loop, the
  loop now takes the cold stream's whole 100 kW of steam heating from the
  other process's hot stream. A loop's heater and cooler are named
  ``Loop <hot>-><cold>:heater`` and ``:cooler``, with ``\`` and ``>`` in a
  stream name escaped by a backslash, so streams whose names hold ``->``
  cannot give two loops one name.
- The fit to every period screened a segmented utility at a stream end with
  its largest segment ``dt_cont`` at both ends, so a utility whose terminal
  segments keep the approach was dropped or replaced by a dearer one when
  another of its segments had a larger contribution. Each end is now held
  to its own terminal segment's contribution (the inlet segment at the
  utility's inlet, the outlet segment at its outlet), as the fit's approach
  equations do.
- The same screen skipped segmented process streams: it took a stream to
  run in a period only when it had a heat capacity flowrate, which a
  segmented stream does not, so any utility passed, and a segmented stream
  could be given the cheapest end utility even when it could not reach the
  stream's target. A segmented stream now runs where its segments carry
  duty, and is screened like any other.
- One network could carry two topology signatures when two matches that
  share no stream were numbered in either stage order, so the ranking and
  the run record listed it twice. On the four-stream Yee and Grossmann case
  at width 1 and depth 1, ranks 1 and 2 were one network (from the 1.0 and
  1.5 multipliers). Signatures now follow each stream's order of
  exchangers, so such a network is listed once and its seeded tasks are not
  repeated; the tasks keep their seed's stage order. Record keys of
  affected networks change.
- A fixed-structure solve (``optimise_duties``, the fit to every period or
  an overdesign rating) that Ipopt stopped at ``max_iter`` or
  ``max_wall_time`` was returned as a plain success with empty diagnostics,
  at a cost that was not a converged optimum. Its outcome now records
  ``diagnostics["unconverged_termination"]``; a verified allocation is kept
  with a ``UserWarning`` to raise the limit in ``HENS_SOLVER_OPTIONS_EVM``,
  and one that fails verification fails with an error that names the
  termination and gives that advice, so the ``WorkflowContractError``
  repeats it. The advice to raise a limit is given only for an iteration,
  evaluation or time limit (``maxIterations``, ``maxEvaluations`` or
  ``maxTimeLimit``); a solve that stopped for another reason, such as
  ``minStepLength``, is named without it, here and in the synthesis
  error that lists the ``HENS_SOLVER_OPTIONS_*`` limits.
- ``optimise_duties(objective="rating")`` rated at 1 K when no
  ``min_approach_temperature`` was given, so it failed on the default
  design's own network whenever evolution had closed an approach below 1 K.
  It now rates at the approach the network was designed to, as overdesign
  does. The docstring now lists four objectives and says rating needs an
  installed area on every heater and cooler too.
- A failed fixed-structure solve (a rating, a fit or ``optimise_duties``)
  was reported as ``network_evolution_method(dt_cont x1)``; it is now named
  as a fixed-structure duty optimisation with its objective.
- Retrofit networks, the rated existing network included, failed
  ``verify_network_feasibility`` with areas 1 to 2 % short: retrofit rates
  each installed unit at its area widened by ``area_tolerance`` and by the
  Chen cap's rating allowance but did not record it. Each network now
  records that widening in ``source_metadata["rating_area_allowances"]``,
  as a rating does, and every frontier network of the guide's example
  verifies.
- A total site design with ``levels="placed"`` carried utility
  placement's fallback ``HU`` and ``CU`` into the design: on the unit test
  site, two levels sent all cooling to a -30 C cooler priced as cooling
  water, and three levels added a 300 C heater as a fourth level. The placed
  case now keeps the case's own utility that placement priced the fallback
  as (here cooling water at 20 to 25 C, or HPS), and a count fails as
  infeasible when the case has no such utility. Placed levels closer than
  1 K at every temperature are one level: the case keeps the first. When a
  multiperiod placement falls back at a different temperature in each
  period, the case keeps the utility nearest each period's fallback, at
  that period's utility temperatures.
- A site network could not be designed when a process stream held two
  utility levels between its recovery matches (for example a hot stream
  raising steam on two placed mains), as the placed steam-main site does
  with ``levels="placed"``. A stream runs at most one staged heater or
  cooler at a stage boundary, and the site seed dropped the empty stages
  that gave the second one room. Site seeds now carry each process's staged
  heaters and coolers, and the site's evolution tasks keep a stage boundary
  for each.
- Heuristic-only pinch design (width 0, depth 0) took 21 minutes on the
  packaged ``pulp_mill`` case. Almost all of it (1,141 of 1,180 s) was
  the below-pinch side: each of its 50 proposals was solved holding the
  utility levels, failed, and was solved again without them. At search
  depths 0 and 1, a case of more than 30 process streams now gets fewer
  heuristic networks and side proposals per multiplier when
  ``HENS_HEURISTIC_PDM_NETWORKS`` is unset: 50 times (30 / streams)
  squared, at least 10. ``pulp_mill`` (64 streams) gets 10 and designs in
  about 4 minutes, with a total annual cost 0.1 % above the 50-network
  design's. A heuristic task also stops at a 300 s wall-clock limit: past
  it, each side stops at its first realised proposal and the joining at
  its first network, and a warning says the task kept the networks found
  so far (``heuristic_report["time_limited"]``). ``chocolate_factory``
  (65 streams) also gets 10; cases of up to 30 streams, including the
  other sample cases, are unchanged.
- Overdesign rated a network the fit to every period had fitted at the
  1 K fallback at the design approach, which its 1 K areas cannot meet, so
  ``overdesign_percentage=10`` raised ``WorkflowContractError``. A fitted
  network is now rated at the approach its fit held.
- ``overdesign_percentage="optimise"`` no longer drops a network when the
  rating at a positive margin fails: the search leaves that margin out,
  records it in the new ``NetworkOverdesign.failed_margins``, and after a
  failed maximum searches up to the largest margin that rates. When no positive margin could be
  rated, the result keeps 0 % and a ``UserWarning`` says so, rather than
  reporting 0 % as the optimum.
- **Breaking:** an overdesigned network carries its margin in a typed
  ``overdesign`` field (``NetworkOverdesign``), no longer in
  ``summary_metrics["overdesign"]``. Read ``selected_network.overdesign
  .percentage`` (and ``periods``, ``cost``, ``cost_without_margin``,
  ``search``) in place of the old dictionary keys. Overdesigned results now
  reload from JSON; as a nested dictionary in the scalar summary metrics
  they failed validation.
- A rating (``optimise_duties(objective="rating")``, overdesign) is
  verified with each unit's rating allowance: the cap is calibrated with
  Chen's LMTD at the design point, so off it a unit can carry a few per
  cent more duty than exact sizing allows, which verification used to
  reject, dropping the network. The largest allowance is recorded as
  ``summary_metrics["rating_area_allowance"]``; installed areas are
  reported unchanged.
- A credit (negative) price for a cold utility level no longer makes a
  total site design fail. The credit paid for any steam dumped, so buying
  steam, letting it down and dumping it earned without limit; a level with
  a credit now dumps no more than the processes raise on it. Input
  validation no longer warns "Value should be non-negative" for a cold
  utility's negative price; a negative price on a hot utility, or on a
  segmented or profiled cold utility, still warns.
- A total site whose steam credits exceed its purchases can be designed.
  Its net utility cost, and so its total annual cost (the evolution
  objective), is negative, and every solve with a negative objective was
  marked failed, so ``total_site_heat_exchanger_network`` raised (the unit
  site with LP steam raised at -200 $/MWh). A model that costs a site
  utility balance now accepts a negative objective; other models still
  reject one. ``HeatExchangerNetwork``, the network input contract and the
  synthesis method output and result accept a negative (finite)
  ``objective_value``, ``total_annual_cost`` and ``utility_cost``;
  ``capital_cost`` stays non-negative. A site level move whose net utility
  cost came back at solver noise below zero (-1.1e-05 $/y) is no longer
  logged as failed.
- Heat pump retrofit targets price each side at the utilities the
  network's heaters (or coolers) use, duty-weighted, not at the first one
  listed; a dearer utility listed first overstated the operating cost.
- A Carnot heat pump in a retrofit pays ``new_unit_cost`` for its
  evaporator and condenser, as any new exchanger does.
- ``design.pinch_design``, ``design.thermal_derivative`` and
  ``design.network_evolution`` refuse a ``period_id`` on a multiperiod case.
  They design over every period and used the period only as a label, so
  the stored targets described one period and the design all of them. To
  design one period, use ``heat_exchanger_network(design_basis="period",
  period_id=...)`` or run the method on ``single_period_case``.
- A standalone ``design.pinch_design`` of a case with only hot or only cold
  streams returns its network of heaters or coolers, as
  ``heat_exchanger_network`` does; before, it raised.
- A total site design no longer loses its level-held seed when a process
  raises steam from a hot stream's hottest part ahead of its recovery
  matches (steam raising limited by the main's temperature). The seed's
  stages were compacted so that its first recovery match sat in stage 1,
  where no staged cooler can precede it, and its first model was
  infeasible; only the lowest-cost seed was evolved and the site imported
  all its steam (30,281 against 11,031 $/y in one case). A site seed's tasks
  now keep an empty stage at each end.
- A fitted, rated or overdesigned network that does not hold its design's
  utility levels keeps them as its targets, so ``holds_utility_levels``
  checks them. Before, the targets were dropped and every such network
  reported holding its levels.
- Duty optimisation, rating, overdesign and the fit to every period keep a
  network's utility slices. Pinch design slices a sliding utility (cooling
  water, hot water) at the streams it must reach, and names each slice in
  the exchanger id; these tools ran each heater or cooler over its whole
  parent utility instead, so a slice only part of the range could reach
  was rejected ("on the wrong side of the stream's temperature range") and
  the network could not be rated, overdesigned or fitted.
- Overdesign no longer drops the cheapest networks. It rated every network
  at the 1 K default approach, while evolution designs down to 0.1 K, and
  rated each unit at exactly its design area, which left the solver no
  slack; so even a margin of 0 could make the cheapest network infeasible
  and select a dearer one (758,193 against 720,517 $/y in one case). Rating
  for overdesign now uses the approach the network's method designed it
  to, and the rating cap has a 1e-6 relative slack.
- A placed site design (``levels="placed"``) is no longer cached as the
  problem's design: its networks name the placed utilities (``iso_1`` and
  so on), which the problem lacks, so later tools failed on them.
- Stage reduction keeps a network with fewer stages only when it costs no
  more and uses no more hot or cold utility in any period than the network
  it reduces. Before, it kept any reduced network that solved and verified,
  so a pinch design network could quietly lose recovery. The reduced solve
  keeps the task's time limit; a nonlinear reduced model, solved with
  Ipopt, takes it as ``max_wall_time`` and no other option of the task's
  solver. A layout of more than 10 stages now drops stages under 10% of the
  recovery, as one of 9 or 10 does, and empty stages are dropped even when
  no stage carries enough of the recovery to meet its threshold; before,
  every stage was kept, empty ones included.
- Stage packing (``HENS_STAGE_PACKING``, and the minimum-unit pinch
  design search) counts each match's duty in every period. Before, it read
  the first period alone, so a match that carries load only in later
  periods was forced off and a packed multiperiod MILP pinch design could
  fail.
- Network evolution ranks a match for removal by the utility cost its
  recovery saves in every period, averaged by the period weights. Before,
  it used the first period's duty and price, so a match that recovers heat
  only in later periods looked like pure capital and was removed first.
- The fit of a design to every period (the default for multiperiod
  designs) keeps every exchanger, in every period, at the approach the
  network was designed to: a pinch design network its streams' temperature
  contributions at its own dt_cont multiplier (7.5 K at 0.75x, not the
  reference 10 K), an evolved network the 0.1 K floor (or
  ``HENS_MIN_EXCHANGER_APPROACH_SHARE``). Before, it fitted at 1 K, so a
  default multiperiod design could sit below ΔTmin (8.1 K against 10 K in
  one case) and still verify. Stream-end heaters and coolers are padded at
  the same approach, each pair's contributions times the multiplier, so a
  utility that cannot keep it is not chosen only to fail the fit; on the
  packaged ``dairy_chilling_multiperiod`` case the default design keeps its
  approach. Pinch design networks at different multipliers are no longer
  fitted once and shared. Where the approach cannot be met, the fit warns
  and falls back to 1 K.
- The fit to every period holds a design's utility levels where it can.
  A network that holds its levels keeps each level at its target in every
  period, and records the targets. Before, the fit dropped them, could move
  load from one level to another by price (LPS to HPS), and still reported
  the levels as held. When a period's targets cannot be met exactly (as for
  a design on the median period), the fit warns and releases the levels.
- A stage-wise network whose matches split a segmented stream in one stage
  sizes each match on its branch's share of the flow. Before, each branch
  was sized on the whole stream's profile, which ended it short of the
  stage outlet and understated its area (by 30% in one two-branch case);
  the same profile skewed the dQ/dA rankings of the thermal derivative
  method and evolution, and the extracted area slices. Evolution models,
  whose branch outlets may differ, use the solved flow split; with
  isothermal mixing each match's share of the stage duty is its flow
  share. A branch with a very small share keeps a valid profile.
- A multiperiod pinch design (MILP or heuristic, design basis ``"all"``)
  no longer fails when a stream has no flow in one period. Its matches
  carry no duty in that period; before, joining the two sides of the pinch
  raised "Cannot set a disallowed match's constant" and no network was
  designed. A match counts as idle in a period only below the side's own
  solve tolerance, so a small active duty keeps its approach temperatures.
- ``design.retrofit`` no longer defaults ``min_approach_temperature`` to
  1 K, which let new matches recover heat the design rules out and
  overstated savings. Without it each exchanger keeps the sum of its two
  sides' temperature contributions, so per-stream ``dt_cont`` and zone
  multipliers hold; new matches and heat pumps are screened at twice
  ``THERMAL_DT_CONT``. A given value still applies to every exchanger. A
  retrofit that prices new units but leaves ``area_addition_cost``,
  ``split_cost`` and ``repiping_cost`` at 0 now warns, since the search
  then favours those modifications.
- A retrofit no longer stops when a real-fluid heat pump's cycle cannot be
  solved at its duty to price the machine; that candidate is dropped and
  the search goes on.
- A retrofit whose existing network names a stream without its zone
  (``H1``) when several zones have a stream of that name (``A.H1`` and
  ``B.H1``) now raises a ``ValueError`` naming them, also when another
  exchanger names one of them in full. Before, both mapped to ``H1`` and
  one stream's temperatures and heat capacity flowrate silently stood for
  the other's in the move screens and heat pump targets. A name one stream
  has exactly (``H1``) no longer also stands for ``B.H1``.
- Two networks that differ only in where a heater or cooler sits among the
  recovery stages (or a staged against a stream-end unit) are no longer
  ranked as one structure, which dropped one of them as a repeat. Utility
  stages are renumbered with the recovery stages, so the same grid with its
  stages shifted is still one structure.
- A multiperiod stage-wise model no longer starts infeasible when a match
  that can run in one period cannot run in another (its supply temperatures
  there are closer than its approach): in such a period the match carries
  no duty and has no approach or minimum dQ/dA constraints. A multiperiod
  warm start also
  leaves a heater or cooler that evolution removed switched off, instead of
  charging its unit cost at no duty.
- The median design basis (the default for multiperiod designs) works when
  streams are given by heat capacity flowrate rather than heat flow, including
  streams in nested zones and same-named streams in different zones. Before,
  it read no heat flow, found no operating period and raised.
- A multiperiod network from duty optimisation (``optimise_duties``, a fit
  to every period, overdesign rating) now records each period's stream
  temperatures, duties and heat transfer coefficients, so verification
  checks every stream's heat balance in every period. Before, it found no
  per-period data and passed a stream left short of its target. The area
  check also uses each period's coefficients rather than the first
  period's.
- At search depth 2, a MILP pinch design network none of whose thermal
  derivative tasks succeeded is now evolved directly. Before, only a
  missing Couenne for every task sent the MILP networks to evolution; any
  other failure (a crash, a time limit, a failed check) left them as
  unevolved final candidates with no warning.
- ``HENS_TDM_PARENT_LIMIT`` now limits how many MILP pinch design networks
  the thermal derivative step refines, cheapest first; before, it was only
  recorded. A network the limit leaves out is not sent to network evolution
  in place of its derivative tasks (only one whose derivative tasks ran and
  failed is evolved directly); below search depth 3 it is a final candidate
  as it stands. A repeated ``HENS_DERIVATIVE_THRESHOLDS`` value no longer
  solves the same task twice.
- A design run no longer stops with a ``TypeError`` when a heuristic pinch
  design network has no objective value; it counts as the dearest. When one
  solve in a parallel stage raises, the solves still waiting for a worker
  are cancelled instead of all being run before the error is reported.
- Each thermal derivative task now stops at 300 s unless
  ``HENS_SOLVER_OPTIONS_TDM`` sets a time limit (``time_limit`` for
  Couenne, ``limits/time`` for SCIP). Before, a depth-2 design could run
  for hours in Couenne. A task stopped there keeps its best network when it
  has one and that network verifies; otherwise the task fails.
- ``design.total_site_heat_exchanger_network(width=, depth=, options=)``
  designs one network for a site of several process zones. Each process is
  designed alone with the pinch design method, its utilities placed by
  temperature so each level carries its targeted duty; the process networks
  are joined, one per process at each dt_cont multiplier, and network
  evolution refines them as one problem with no thermal derivative step.
  Every process keeps each utility level at its duty, and heat is recovered
  only between streams of one process. On ``zonal_site`` every process meets
  each of its level targets. ``design.heat_exchanger_network()`` on a problem
  with more than one process zone now emits a ``UserWarning`` that it pools
  every process's streams (any inter-process match, no piping cost) and
  names the site method.
- ``design.total_site_heat_exchanger_network`` pays only for the site's net
  utility use. The network evolution model of a site carries the site
  utility balance: per period each utility level (a boiler, a steam main or
  hot water circuit, cooling water) closes with imports, letdown from a
  hotter level and dumps, and the model costs those instead of pricing each
  heater and cooler. Steam a process raises on a main that another process
  uses is no longer charged as a cooler. A ``Both`` utility is one main; a
  hot and a cold utility that only share a name stay apart; letdown follows
  each period's temperatures; and a hot utility's ``maximum_heat_flow`` caps
  what the site buys into its level. Each site network carries
  ``site_capital_cost``, ``site_operating_cost`` and
  ``site_total_annual_cost`` in its summary metrics and the balance in
  ``source_metadata["site_utility_balance"]``; the record and the result
  rank site networks by site total annual cost.
- ``design.total_site_heat_exchanger_network`` designs each process twice,
  with its utility levels chosen by price and held at their targets, and
  seeds network evolution from both: at each dt_cont multiplier a
  lowest-cost and a level-held site seed, each taking every process's
  fewest-unit network (then the lowest total annual cost). The site design
  may now select a network whose levels follow price where that is cheaper
  for the site; the level-held networks stay among the ranked results.
- ``design.total_site_heat_exchanger_network`` works on multiperiod sites:
  each process is designed over every period at once, so each level carries
  its targeted duty in every period, and network evolution holds those
  duties period by period with recovery only within a process. On
  ``zonal_site_multiperiod`` every process meets each level in each period.
- ``design.total_site_heat_exchanger_network(levels="placed")`` sweeps the
  number of site utility levels. For each count in ``level_counts``
  (``HENS_SITE_LEVEL_COUNTS``, default 2 to 5) utility placement places that
  many levels, each serving heating and cooling, with the case's sensible
  utilities kept sensible while two isothermal levels remain; the site is
  designed with each set and the cheapest design is selected. A count whose
  placement is infeasible or whose design fails is recorded and skipped; the
  result ranks every network of each designed count (so the level-held
  networks stay among them) and its provenance lists each
  count (``HENS_SITE_LEVEL_SWEEP``). ``HENS_SITE_LEVEL_EARLY_STOP`` ends the
  sweep after two counts in a row that cost more. ``levels="given"``
  (``HENS_SITE_LEVELS``, the default) keeps the case's utilities.
- A placed total site design (``levels="placed"``) refines the cheapest
  count's design by level-temperature moves: each placed level is shifted up
  or down by ``HENS_SITE_LEVEL_STEP`` (default 5 K), within the order of the
  levels and the site's stream temperatures, each process's level duties are
  re-targeted, and the site is evolved again from the current best network;
  a move is adopted only on a lower site total annual cost than the
  unshifted levels evaluated the same way (the sweep's best network can
  come from a seed a move never builds). The result is the cheapest of the
  sweep's network, the unshifted levels' re-design and the adopted moves;
  when the sweep's network is kept over adopted moves, the log ends with a
  ``"start kept"`` entry. The step halves when no move improves, down to
  ``HENS_SITE_LEVEL_STEP_MIN`` (default 1 K); ``HENS_SITE_LEVEL_STEP: 0``
  turns the moves off. Every move tried is listed in the result's
  provenance (``HENS_SITE_LEVEL_MOVES``). Each set of level positions is
  designed once: a move back to positions already designed is logged
  ``"skipped"`` with the cached cost, or with ``"already failed:
  <message>"`` when their design failed. Moves refused for meeting another
  level or leaving the stream temperatures are logged ``"refused"`` with
  the reason.
- ``HENS_SITE_TIME_LIMIT`` (seconds, default no limit) bounds a placed
  total site design: once it runs out no further level count or level move
  is started, and the cheapest design so far is returned. The first count
  is always designed; skipped counts are recorded ``"not run"`` with
  message ``"time limit"``, and the moves end with a ``"stopped"`` entry.
  The limit does not apply to ``levels="given"``: that is one design with no
  point to stop between, and its solves are bounded by the synthesis solver
  time limits. A given-levels design warns when ``level_counts``,
  ``placement`` or ``HENS_SITE_TIME_LIMIT`` is set, since it ignores them.
- A total site seed no longer takes a process network designed at a
  smaller dt_cont multiplier than its own: such a network can need more
  recovery than the seed's approach allows. Each process gives its nearest
  network at or above the seed's multiplier, and there is no seed at a
  multiplier some process has nothing at or above. A process zone whose
  pinch design found no network is named in the error.
- A level-held total site seed no longer takes a process's level-held
  network from a smaller dt_cont multiplier when only its lowest-cost
  networks reach the seed's multiplier; that process gives its lowest-cost
  network instead.
- ``design.optimise_duties``, ``design.retrofit`` and
  ``design.retrofit_heat_pumps`` refuse a total site network (a site seed or
  the network ``total_site_heat_exchanger_network()`` returns) with a
  ``ValueError``: their fixed-structure models neither hold each process's
  utility levels nor keep the site utility balance, so they re-priced each
  heater and cooler alone (31,941 $/y against a site cost of 21,880 $/y in
  one case) and dropped the levels, and retrofit proposed matches between
  processes the site model forbids. A stored
  ``HENS_OVERDESIGN_PERCENTAGE`` no longer reaches a site design's process
  designs, where rating each network at the margin dropped its level
  targets and turned the level-held seeds into duty-based ones.
- A placed site level is one main: its hot and cold utility share one name
  (``iso_1`` rather than ``hot_iso_1`` and ``cold_iso_1``), so site utility
  accounting credits steam raised on a placed level against its use.
- A total site network design no longer fails when one process's pinch
  design with utility placed by price fails: that process seeds evolution
  from its level-held networks (and the reverse), and the design fails only
  when both fail for a process.
- Network evolution over several periods no longer rejects every network it
  evolves: verification now accepts a period whose approach is wider than
  the model's approach variable (the shared area is set by another period,
  with bypass in this one), leaves out the area of a match evolution
  removed, and no longer turns an unused area a hair below zero into NaN.
  Multiperiod designs at depth 1 and above can now select an evolved
  network instead of falling back to the pinch design.
- A network evolution or stage-wise model can hold a utility level's duty
  over a group of streams and allow recovery only within a group; a stream
  with several stream-end heaters (or coolers) is now reported as split
  between them, as the model sizes them, so such networks pass the
  exchanger heat balance checks. The recovery area check now allows for the
  model's own area smoothing, so a tiny match is no longer rejected for it.
- ``design.retrofit_heat_pumps`` gives an existing network's ideal
  and as-piped heat pump targets from the HPR service: the ideal against
  the process's heating and cooling targets, the as-piped against only the
  existing heaters and coolers (no process-side repiping). Each reports
  its levels, work, COP and the utility and operating cost before and
  after; ``network_gap`` and ``heat_pump_gap`` split the difference
  between them. ``compute_profile_heat_pump_target`` in the HPR service
  targets a heat pump against any given source and sink profiles.
- Extracting a network from a solved model is cheaper: plain numbers,
  NumPy floats and ``[value]`` lists are read directly rather than through
  Pyomo lookups and a caught ``TypeError``, and recovery extraction looks
  up each period's duty grid and each match's row once rather than for
  every stage. Under a profiler, a design on the pulp mill sample case
  spends about 28 s in extraction instead of 34 s.
- ``design.heat_exchanger_network(zone=...)`` designs a network for one
  zone, given by name as in targeting or as a ``Zone``, for example
  ``site.design.heat_exchanger_network(zone="Hospital[S]")`` or
  ``site.design.heat_exchanger_network(zone=site.master_zone.subzones["Hospital[S]"])``. The network
  serves that zone's streams alone, with every utility of the problem
  (including any default utility, placed for the whole problem), and
  the problem's stored results are not changed. The default, the root zone,
  designs on every stream as before.
- ``design.heat_exchanger_network()`` now designs a problem or zone with
  only hot or only cold process streams, which used to fail with "prepared
  PinchProblem must contain hot and cold streams". Its network is heaters
  (or coolers) alone, and each utility level takes the duty targeting gives
  it: the least valuable level serves the composite curve from the pinch end
  first, so a stream spanning several levels gets utility exchangers in
  series. The network is built without a solver and serves every period;
  overdesign is not available for it.
- With more than one hot or cold utility, the pinch design method (MILP and
  heuristic) now holds each utility level at the duty targeting gives it,
  rather than choosing levels by price. To place a level where it serves,
  the stage-wise model can put a heater or cooler between stages, in series
  between them; such units are reported as
  ``hot-utility:<utility>-><stream>:S<stage>`` (or
  ``cold-utility:...:S<stage>``), with ``stage`` the stage the stream has
  just left, and the grid view draws them on that stage boundary. A side
  that cannot hold its levels (for example hot water, whose fixed outlet
  can face a stream entering above it) is solved as before, choosing levels
  by price, with a warning. The heuristic method designs the levels into
  its proposals: each level with a target is matched like a process stream
  that must be used, and a side is split at its utility pinches (where,
  with the levels at their targets, the heat cascade closes) and each
  region designed on its own, so no match crosses a utility pinch. A
  heuristic side holds each level at its target or, failing that, within
  0.1% of it, the share of a load the heuristic counts as met. Networks
  that hold the levels (each utility within 0.1% of its target) rank
  ahead of those that do not, then by total annual cost, so the selected
  network holds the targeted levels whenever one does. The later stages
  (temperature-driven method, network evolution) do not yet hold the
  levels, so their networks usually rank behind. A network that holds levels keeps its stages: stage reduction
  re-solves by utility cost, which would undo them.
- A utility whose temperature slides, such as hot water from 90 to 70 C,
  now reaches the streams targeting lets it heat. A heater runs its utility
  over the whole range, so hot water could not heat a stream entering at
  82 C; with levels held, such a utility is cut into slices at the streams'
  supply temperatures (plus their approach), run in series, each taking its
  share of the level's duty. Their exchangers are reported under the
  utility's name, with the slice in the exchanger id (for example
  ``hot-utility:Hot Utility.HW@90-87.1->...``). On ``zonal_site`` the
  Hospital now holds its hot water and steam levels.
  ``HENS_UTILITY_SLICES: false`` turns the slices off.
- ``HENS_UTILITY_PLACEMENT`` chooses how synthesis shares duty between
  utility levels: ``"temperature"`` holds each level at the duty targeting
  gives it, lowest exergy first, as a site design needs; ``"price"`` lets
  the pinch design choose levels by utility cost, as before. The default,
  ``"auto"``, is ``"price"`` for a design covering one process zone (an
  isolated zone, such as ``heat_exchanger_network(zone="Hospital[S]")``)
  and ``"temperature"`` for one covering several.
- The pinch design method no longer drops a side of the pinch whose
  utility target is zero but whose streams still exchange heat there, as in
  a threshold problem. Before, that side's matches were left out of the
  network, which then failed verification.
- Network evolution ranks the matches it could add by their dQ/dA
  averaged over the periods by the period weights, as it already weighted a
  match's recovered value. Before, a multiperiod evolution read period 0's
  approaches and temperatures only, so a match that helps mostly in a later
  period ranked low. Single-period runs are unchanged.
- A network evolution task whose initial model fails now raises a
  ``UserWarning`` naming the task, once per task. Before, the evolution was
  skipped with only a log line, often in a worker process.
- ``HENS_SITE_LEVEL_HOLDS`` sets how network evolution of a total site
  network design holds each process's utility levels: ``"fixed"`` (the
  default, as before) at the duties its pinch design targeted, or
  ``"at_least"`` at or above them, so load may shift between processes
  under the site utility balance. The ranking still puts networks that
  hold each level at its target first.

Testing and CI
~~~~~~~~~~~~~~

- The solver tests now run on every pull request into ``develop`` and must
  pass for "CI OK"; before, they ran only for pull requests into ``main``.
  The packaged HEN notebook runs, about 8 of the lane's 14 minutes, still
  run only for pull requests into ``main`` and releases.

Documentation
~~~~~~~~~~~~~

- The HEN synthesis guide and the ``PinchProblem`` API page cover
  ``design.retrofit_heat_pumps`` and ``design.design_basis``. The API page
  notes that ``retrofit_heat_pumps`` stores the first period's direct
  target when none is cached and needs the HEN solver to rate a network
  whose heaters or coolers lack solved temperatures; the guide notes that
  the ``"rating"`` objective needs an installed area on every exchanger,
  heaters and coolers included.
- Corrected HEN docs that no longer matched the code: the placed level
  sweep ranks every network of each designed count, not each count's best;
  ``level_sweep_view`` and the application README no longer say the sweep
  is cached as the problem's design; the API page says ``design.retrofit``
  caches the direct target when none is cached; ``retrofit_heat_pumps``
  says its prices are duty-weighted over the network's utilities; and
  ``top(n)`` says it returns ranked outcomes, each holding its network.
  The guide notes that a cold utility may carry a credit price.

0.9.3 (2026-11-18)
------------------

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Faster ``design.heat_exchanger_network`` with an overdesign margin or on
  a ``"median"`` or ``"period"`` design basis: a network the search offers
  again from another task is fitted and rated once, not each time. On the
  four-stream Yee and Grossmann case at 10% overdesign this rates 23
  networks rather than 60.
- Post-processing a solved stage-wise model is cheaper: the alpha flow-on
  solve now runs only when evolution ranks the model's matches, so pinch
  design variants and evolution candidates that are never ranked skip it.
  On the pulp mill sample case a design spends about 6 s in
  post-processing instead of 14 s, with the same network.
- Building the amalgamated pinch design network is cheaper: it holds
  values only, so its constraint expressions are no longer built and
  dropped, and stage-wise builds look up each match's approach temperature
  once rather than for every stage. On the pulp mill sample case
  amalgamation takes about 5 s of a design instead of 8 s, with the same
  network.
- ``design.retrofit`` takes ``repiping_cost`` ($ per stream): each new
  match, relocation and new end heater or cooler also costs that for every
  stream whose piping it changes, and each modification reports those
  streams in ``repiped_streams``. A relocated exchanger keeping its stream
  in the same stage does not repipe it.
- ``design.retrofit`` can add heat pumps (``heat_pump_cycle``): an
  evaporator at the end of a hot stream and a condenser at the end of a
  cold one, rated by the HPR service's Carnot cycle or a single real-fluid
  vapour compression stage on ``HPR_REFRIGERANTS``, with work at
  ``COSTING_HPR_ELE_PRICE``. A Carnot heat pump's compressor costs
  ``heat_pump_cost`` per kW; a real-fluid one is priced with the HPR
  service's installed machine capital. Modifications of kind ``heat_pump``
  report ``heat_pump_cop``, ``heat_pump_capacity`` and
  ``heat_pump_refrigerant``; ``FixedStructureSpec.heat_pumps`` links a
  heat pump's evaporator and condenser duties by its COP. A heat pump whose
  condenser or evaporator name (``HP H1->C1:condenser``) a stream or
  utility of the problem already has is not offered, so that stream or
  utility keeps its own properties.

0.9.2 (2026-11-17)
------------------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A registered cycle can name a model family of its own as a string
  topology family, with no new ``HeatPumpAndRefrigerationCycle`` member:
  ``register_cycle`` makes it a valid ``HPR_TYPE``, and
  ``target.heat_pump_and_refrigeration`` runs it. Families compare by
  ``HPR_TYPE`` value (``family_value``).

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed: a fixed-structure solve (``optimise_duties``, retrofit) could be
  infeasible when a segmented stream's heat capacity falls along it, as
  the duty bounds used the exact total while the balances use the
  rounded profile. Verification now accepts a segmented exchanger with
  more area than its duty needs, as it already did for single-segment
  exchangers, so an existing network with segmented streams can be rated.
- Fixed: ``design.retrofit`` no longer stops when a move leaves a structure
  the model cannot build (streams with no utility exchanger whose loads
  cannot balance); the move is dropped. Verification now allows an
  installed exchanger the extra duty the model rates it for off its design
  point, so a multi-period network OpenPinch designed rates again. The
  default search ranks at most ``ceiling_relocations`` (default 400)
  relocations per design; the 72-exchanger pulp mill offers 14,617.
- Fixed: with ``HENS_MAX_PARALLEL`` above 1, each task's network enters a
  ``HeatExchangerNetworkRecord`` as soon as its task finishes, not after
  slower tasks submitted before it.
- New ``HeatExchangerNetworkRecord``
  (``OpenPinch.analysis.heat_exchanger_networks.results.record``), passed as
  ``design.heat_exchanger_network(record=...)``: a thread-safe record that
  receives every network as soon as it is verified and costed in its final
  form, for another thread to poll with ``version``, ``snapshot()`` and
  ``changed_since(version)``. Worker processes send their networks to the
  calling process. See "Networks as They Are Found" in the HEN synthesis
  guide; tutorial 16 follows its deeper search this way.
- A synthesis result now ranks every verified network of the run, including
  each improvement an evolution search finds while it runs, not only the
  network each task ends with. A network enters only with a converged solve,
  a total annual cost and a passing feasibility check, so a successful
  outcome without a total annual cost is no longer accepted.
- A multiperiod design on a median or period basis now fits every network to
  every period, and an overdesign margin rates every network, before it is
  ranked; the cheapest fitted (and rated) network is selected, which can
  differ from fitting the cheapest basis network alone. These designs take
  longer, since each network costs a fit and, with a margin, its ratings.
- New ``design.retrofit(existing, ...)`` finds cost-effective modifications
  of an existing network. The existing network, with every exchanger's
  installed area, is rated, then changed one structural step at a time, as
  in network evolution: a new match (joining a stage or in a new one) or,
  with ``relocation_cost``, an existing recovery exchanger moved with its
  area to another match or position. Following the network pinch approach
  of Smith's group, candidate changes are ranked by the utility their
  structure reaches with unlimited area, and the best are solved on the
  fixed-structure model with a new ``"retrofit"`` goal that also sizes the
  added area each structure needs (existing area sunk; added area, new
  units, area additions and splits priced). New matches are screened by the
  utility they could cut at first order. With ``new_utility_units=True`` a
  change may come with the new end heater or cooler that frees the units it
  needs. ``max_added_area_fraction`` caps the area added to an existing
  exchanger. ``enhancement_factors``, ``enhancement_cost`` and
  ``enhancement_area_cost`` add heat transfer enhancement (tube inserts)
  of existing exchangers as a move, checked against Akpomiemie and Smith
  (2018); ``enhancement_pressure_drops`` and ``pressure_drop_allowances``
  keep inserts within each stream's pressure drop allowance. With Ipopt the stepwise search adds area only on exchangers whose
  area is worth something, from the solver's duals; the fixed-structure
  model reports them as ``area_values`` and has a ``"utility cost"`` goal.
  ``HENS_MAX_PARALLEL`` runs each step's solves on threads.
  Every design is checked as optimised duties are (approaches, heat
  balances, areas). The result's ``frontier`` gives the best design
  for each number of structural changes with savings, capital, payback and
  net annual benefit, and marks dominated ones; ``strategy="stepwise"``
  instead counts each area addition as a step. Network mappings for
  ``optimise_duties`` and ``retrofit`` now take entries as mappings with
  ``area`` and ``id``. Demonstrated in notebook 16 on the Ciric and Floudas
  (1989) case, now packaged as ``Five-stream-Ciric-and-Floudas-1989.json``,
  and checked against Akpomiemie and Smith's (2018) crude preheat train,
  whose published retrofit targets it reproduces; see "Retrofit of an
  Existing Network" in the HEN synthesis guide.
- The heuristic pinch design method realises networks faster: side
  proposals that solve to a side network already found are not paired,
  since each pairing would repeat a network; they stand in only for a
  pairing that fails. On the
  12 OpenHENS cases at the default width and depth 0 it takes about 23 %
  less time at 10 networks per multiplier, with the same best networks and
  at least as many distinct ones. ``heuristic_report`` counts the dropped
  sides as ``repeated_sides``.
- A heuristic side that leaves a stage empty is no longer dropped as a
  repeat of the same side network without one. Only a pairing with an empty
  stage is re-solved at stage reduction, which can rebalance duties into
  another network: on Four-stream Escobar the dropped side gave a network
  with one heater fewer, the parent of the cheapest evolved network
  (352,267 against 358,065 before evolution explored two branches). On the
  12 OpenHENS cases at the default width and depth the best networks are
  unchanged and the run takes about 1 % longer.
- A heuristic side proposal's solve is limited to 10 s
  (``HEURISTIC_SIDE_TIME_LIMIT``, unless the solver options set a limit),
  and a side stopped there is dropped, as its incumbent need not hold the
  minimum utility. The solve is a MIP over the proposal's matches, and on
  the Six-stream Spray Dryer case at the default width and depth HiGHS did
  not close one in hours, so the design never finished. It now takes 171 s
  (8,256,978 TAC), with 14 of its 404 side solves stopped; on the other 11
  OpenHENS cases every side solve takes at most 0.25 s and the best
  networks and run times are unchanged.
- ``HENS_HEURISTIC_PDM_NETWORKS`` defaults to 50 heuristic networks per
  dt_cont multiplier at search depths 0 to 3 (was 10 at depths 0 to 2 and 20
  at depth 3); depth 4 already used 50. With the faster realisation, the 12
  OpenHENS cases at depth 0 take about 50 s instead of 22 s at 10, give
  about 2.7 times as many distinct networks, and improve the best network on
  four cases. The pulp mill case takes 188 s instead of 45 s, finds 11
  minimum-unit networks instead of 2, and its best network is within 0.01 %.
  Set ``HENS_HEURISTIC_PDM_NETWORKS`` to 10 for the earlier, faster search.
- Network evolution explores two add and two remove branches per step by
  default (``HENS_EVM_N_AD_BRANCHES`` and ``HENS_EVM_N_RM_BRANCHES`` now
  default to 2 at search depths 1 and 2, was 1), and the evolution tasks of
  a serial stage share the topologies they have solved, so a topology one
  task solved is reused by the next rather than solved again. Tasks solved
  in worker processes (``HENS_MAX_PARALLEL`` above 1) do not share. On the
  12 OpenHENS cases at the default width and depth, the best network costs
  less on five cases (Five-stream Bogataj 4.0 %, Thirteen-stream Kim 1.6 %,
  Four-stream Escobar 1.6 %, Five-stream Kim 1.2 %, Ten-stream Ahmad
  0.04 %), each with one to four fewer units, and no case is worse; the run
  takes about as long (+2 %). On the pulp mill, chocolate factory and crude
  preheat sample cases the best networks are within 0.01 % and the runs take
  about 40 % longer. Set both options to 1 for the earlier single-path
  search.
- Post-processing a solved stage-wise model is faster. The alpha flow-on
  system computes each stream's flow-on terms once per stage rather than once
  per partner, and drops zero terms when it builds them. Flow fractions read
  each solved value once. Per-match work whose result was discarded, and a
  utility price recomputed for every match, are gone. On the pulp mill
  sample case a design drops from 76 s to 65 s, with the same network.
- Adopting an evolved stage-wise topology keeps the candidate's solved
  utility and area costs as numbers instead of deep-copying their Pyomo
  expressions, each of which copied the candidate's whole model. On the pulp
  mill sample case its eight adoptions take 0.7 s instead of about 10 s, with
  the same network.
- Building a design stage's tasks copies the prepared zone once, not on
  every read: ``PinchProblem.master_zone`` deep-copies the whole zone tree,
  and each task read it many times for its settings and solver arrays. On
  the pulp mill sample case a design reads it 3 times instead of 139, which
  saves about 6.5 s, with the same network.
- Stage-wise post-processing is faster. Reading a solved Pyomo variable no
  longer tries to index it first; post-processing reads an inactive match's
  approaches not at all and each stream temperature once; and the alpha
  flow-on step builds its fractions and sparse system as NumPy arrays, with
  the same values to the last place. On the pulp mill sample case
  post-processing takes about 11 s of CPU time instead of 32 s, and the
  design about 154 s instead of 181 s, with the same network.
- Building a stage-wise model is faster: parameter blocks are fixed as a
  whole when allocated, the Pyomo module is looked up once per model, and the
  big-M values and duty bounds that multiply binaries are Python floats, so
  Pyomo builds those terms without NumPy's ufunc dispatch. On the pulp mill
  sample case the 276 stage-wise builds of a design take about 37 s of CPU
  time instead of 58 s, and the design about 157 s instead of 175 s, with the
  same network.
- A script piped to Python on stdin (``python - <<EOF``) or run from a
  string no longer crashes network synthesis or multistart optimisation with
  ``BrokenProcessPool`` under the spawn or forkserver start method (the macOS
  default). Worker processes cannot rerun such a main module, so its tasks
  are solved serially with a warning, as for a script with no main guard.
- Extracting a network from a solved model is faster: a solved Pyomo
  variable's value is read before asking what the object is, and a recovery
  match's binary is read once rather than once per period. Extracted
  networks are unchanged.
- Building index grids and writing parameters is cheaper. A grid is built
  with one call per cell rather than a recursive helper call per cell, and a
  parameter that is already fixed is written without being fixed again,
  which ran Pyomo's domain and bound check on every write. On the pulp mill
  sample case a design takes about 124 s of CPU time instead of 130 s, with
  the same network.
- The amalgamated pinch design network is built without constraints or an
  objective. It holds the above- and below-pinch sides' solved values and is
  only read, never solved, so a values-only model (``values_only=True``) has
  the same variables and parameters and refuses to be solved. On the pulp
  mill sample case this saves about 20 s of CPU time per design (107 s
  instead of 128 s, measured before the index-grid change above), with the
  same network.
- Building stage-wise models is cheaper. Variables come from preallocated
  blocks, as parameters already did; model components are no longer given
  names, which were never used; and a disallowed match's duty, approach
  temperatures, binary and branch fractions are plain numbers instead of
  fixed Pyomo parameters, which also makes loading a model into HiGHS
  faster. On the pulp mill sample case a design takes about 85 s of CPU
  time instead of 108 s, with the same network.

0.9.1 (2026-11-16)
------------------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``target.heat_pump_and_refrigeration(cycle, ...)`` targets any registered
  cycle (``"carnot"``, ``"vapour_compression"``, ``"mvr"``; ``"brayton"`` is
  registered but unavailable) through one call: ``duty``, ``topology`` and
  ``placement`` as arguments, and the load, levels, the cycle's own options
  and the search limits as typed models from ``OpenPinch.contracts.hpr``
  (``HprLoad``, ``HprLevels``, the cycle's options model,
  ``HprSearchLimits``) or mappings of their fields. ``request=`` takes all of
  it as one ``HprTargetRequest``, which a client can validate as JSON. It is
  also on ``target.all_periods``. The seven named methods
  (``carnot_heat_pump`` and the rest) now run through it with unchanged
  results.
- ``target.hpr_cycles()`` lists the registered cycles: duties, topologies,
  availability, optional dependencies and the JSON Schema of each cycle's
  options, for building a cycle picker and option form. It is also a
  command.
- The cycles are registered once each in
  ``OpenPinch.analysis.heat_pumps.cycles.registry``: ``register_cycle``,
  ``cycle_spec``, ``registered_cycles``, ``spec_for_cycle`` and
  ``family_handlers``, with ``HprCycleSpec`` holding a cycle's topologies,
  duties, options model and the ``HPR_*`` keys they set, dependencies,
  screening layouts and analysis handlers. A new cycle is its model, its
  options model and one ``register_cycle`` call.
- ``target.all_periods.heat_pump_and_refrigeration(request=...)`` targets
  each period in turn; a request naming another period than the call is
  rejected.
- The named heat pump and refrigeration methods accept
  ``expander_efficiency=0`` (no expander recovery) and
  ``maximum_restarts=0`` (the first start only) again; the gateway's models
  had rejected both.
- A newly registered cycle with screened refrigerants runs once per
  proposed fluid set through the gateway, as the named methods do.
- A multiperiod summary after a heat pump or refrigeration target replays
  it for each period. The replay failed on the cycle type it had recorded;
  a recorded run now leaves out the options its method selects and keeps
  the gateway's cycle, duty, topology and placement, so a parallel or
  indirect run replays as one.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``design.heat_exchanger_network(overdesign_percentage=...)`` (stored as
  ``HENS_OVERDESIGN_PERCENTAGE``) adds a margin to every heat-recovery area
  and rates the network over every period at those areas, with bypass. A
  percentage is applied as given; ``"optimise"`` picks the margin up to
  ``HENS_OVERDESIGN_MAX`` (50 %) for the least total annual cost. The margin,
  each period's recovery and utility before and after, the cost with and
  without it, and the margins tried are in
  ``selected_network.summary_metrics["overdesign"]``.
  ``design.optimise_duties()`` gains the ``"rating"`` objective: least
  utility cost at the network's installed areas, recovery, heaters and
  coolers alike, each of which needs an area. Heaters and coolers keep
  their installed areas under a margin, so a rated period cannot use more
  utility than they carry. An exchanger's ``area`` may now exceed what its
  segment slices need (an installed, overdesigned area); one below them is
  still rejected.
- Call ``options`` set ``HENS_DESIGN_BASIS``, ``HENS_BASIS_CLUSTERS``,
  ``HENS_BASIS_ACTIVE_SHARE``, ``HENS_BASIS_CANDIDATES``,
  ``HENS_OVERDESIGN_PERCENTAGE`` and ``HENS_OVERDESIGN_MAX`` for that call,
  as the stored options do; they were ignored. The all-period fit and the
  rating solves run with the call's options, solver and tolerance, and a
  median or period design's provenance keeps the settings its basis
  synthesis ran with. A margin applies to the fitted network only, never
  also to the basis design.
- ``application.case_derivation.single_period_case(problem, period_id)``
  derives a new problem holding one period's data alone.
- The all-period fit replaces a basis design's end heater or cooler that
  cannot serve its stream in every period with the cheapest one that can.
- Median basis clusters of identical periods each keep their own period, so
  no cluster is left empty.
- The heuristic pinch design method finds the largest feasible duty of a
  match exactly, piece by piece along the streams' profiles. With segmented
  streams, bisection could stop at the end of the first feasible stretch;
  splits away from the pinch now also find duties that need some load
  before their stage keeps its approach, which can add complete proposals.
- A multiperiod ``design.heat_exchanger_network()`` now synthesises on a
  median basis by default (``HENS_DESIGN_BASIS`` ``"median"``), so default
  multiperiod results change: a synthetic single-period case of each
  stream's period-weighted median over the operating periods, whose network
  is then fitted to every period. ``basis_clusters`` (``HENS_BASIS_CLUSTERS``)
  groups the periods by weighted k-medoids first and takes the heaviest
  group's median; ``HENS_BASIS_CANDIDATES`` ``"all"`` tries every group's.
  ``design.design_basis()`` previews the clusters and the synthetic streams
  without synthesis. ``design_basis="all"`` keeps the simultaneous
  all-period design.
- ``design.multiperiod_heat_exchanger_network()`` is removed, and its
  command returns ``Unknown command``. ``design.heat_exchanger_network()``
  now designs multiperiod networks itself: ``design_basis`` (stored as
  ``HENS_DESIGN_BASIS``) sets what the network is synthesised from.
  ``"all"`` designs over every period at once, as the multiperiod method
  did, running all-period targeting first when there is
  none; ``design_basis="all"`` reproduces the old results. A saved design
  from the removed method reads back as ``design.heat_exchanger_network``
  with ``HENS_DESIGN_BASIS`` ``"all"``.
- ``period_id`` on a multiperiod ``heat_exchanger_network()`` is now a basis
  period, and its results change. Synthesis runs on that period alone, then
  the network is fitted to every period: one area per exchanger, the largest
  any period needs, with a bypass where a period needs less, and a heater or
  cooler at a stream end where a period needs one. Before, ``period_id`` only
  labelled the run, which still designed over every period.

- Network evolution runs the ``openpinch-solvers`` Ipopt again when IDAES is
  installed. Looking for Couenne (the default TDM solver) imports IDAES,
  which puts its older Ipopt download first on ``PATH``, so every later
  evolution solve ran that Ipopt instead. The IDAES directory is now a
  fallback after ``openpinch-solvers`` however it reached ``PATH``, and Ipopt
  and Couenne run the executable found when their solve is configured.
  OpenPinch's own IDAES lookup no longer leaves IDAES first on ``PATH``, so
  worker processes, which inherit it, also run the ``openpinch-solvers``
  Ipopt. Ipopt runs as the bundled executable rather than through the
  ``ipopt`` launcher ``openpinch-solvers`` installs, which started Python
  for every solve (about 17 ms each).
- Multiperiod networks converge where Ipopt's restoration phase failed. The
  area cost's offset at zero area rises from 1e-8 m2 to 1e-4 m2, which keeps
  its slope at an idle unit's zero area moderate: every root solve of the
  dairy chilling sample ended "Restoration Failed" at a feasible point, and
  now all ten converge, with the same final network.
- The default (single-path) network evolution solves each topology once per
  search. It cycled through the same add and remove moves and re-solved
  them, including one infeasible removal at every depth; it now reuses the
  first solve. On the twelve literature cases and the dairy sample this cuts
  evolution solves by about a third, with the same final networks.
- Ipopt (``ipopt-pyomo``) now defaults to ``acceptable_tol 1e-4``,
  ``acceptable_iter 5`` and ``max_wall_time 120`` alongside its bound
  pushes, so a solve that stalls near the optimum stops there, and a solve
  that hangs stops after two minutes. Nine-stream's evolution solves that
  stalled short of convergence now converge, with the same networks.
  Together with the changes above, network evolution on the twelve
  literature cases and the dairy sample takes about 30 % less wall time and
  40 % less Ipopt time, every root solve converges (221 of 221, from 211),
  and every final network and its cost are unchanged.
  ``acceptable_iter`` and ``max_wall_time`` need Ipopt 3.14 or later, which
  ``openpinch-solvers`` ships; an older Ipopt first on ``PATH`` rejects them
  and fails its solves. Set any of them in the stage's solver options to
  change it.
- Only the cheapest heuristic pinch design networks seed network evolution:
  ``HENS_HEURISTIC_PDM_EVOLUTION_SEEDS`` per dt_cont multiplier, set by the
  search depth when unset (3 at depths 1 and 2, 20 at depth 3, 50 at
  depth 4). The rest are still final candidates. Previously every heuristic
  network seeded evolution. On the 12 OpenHENS benchmark cases at the
  default width and depth, evolution goes from up to 30 seeds per case to
  up to 9, about half the run time, with the same best network on every
  case. Set the option to ``HENS_HEURISTIC_PDM_NETWORKS`` or more to evolve
  every network as before.
- The heuristic pinch design method searches a pinch side again when it
  finds fewer complete proposals than ``HENS_HEURISTIC_PDM_NETWORKS``: first
  trying the next nearest must streams to the pinch as well as the nearest,
  then splitting a partner away from the pinch when no partner can take a
  stream's load in series. New proposals follow the first search's, so
  existing ones and their order are unchanged. On the
  12 OpenHENS benchmark cases at the default width and depth 0, Ten-stream
  Ahmad at multiplier 1.0 now gets heuristic networks where the MILP stood
  in, and distinct target-meeting topologies rise from 87 to 93 in the same
  time; with 50 networks per multiplier, from 206 to 250 for about 20 %
  more time. The best network is unchanged or better on every case.
- Extracting a network from a solved model is faster: it reads Pyomo values
  directly rather than through ``float()``, which Pyomo disables with an
  error message that is slow to build, and skips matches that carry no duty
  in any period before reading their temperatures. On the pulp mill sample
  case extraction drops from 213 s to 14 s (profiled), and a heat exchanger
  network design from 137 s to 79 s, with the same network.
- Building a heat exchanger network model is faster: its parameters, mostly
  placeholders for disallowed matches, come from blocks of fixed variables
  allocated ahead rather than one at a time. On the pulp mill sample case a
  design takes about 7.5 s less (136 s to 129 s, measured before the faster
  extraction above), with the same network.

0.9.0 (2026-11-14)
------------------

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- HiGHS and Couenne are optional. The ``synthesis`` extra no longer installs
  ``highspy`` or ``idaes-pse``: install ``openpinch[highs]`` for HiGHS and
  ``openpinch[couenne]`` (then ``idaes get-extensions``) for Couenne and
  IDAES's Ipopt. A stage on its default solver (left unset or set to it)
  falls back to SCIP, with a ``RuntimeWarning``, when that solver is missing,
  and a pinch design model HiGHS cannot take goes to SCIP when Couenne is
  missing; any other solver is kept. The ``synthesis`` extra now installs
  ``openpinch-solvers``, which ships Ipopt 3.14.20 (with MUMPS) as wheels
  for Linux, macOS and Windows, so network evolution needs no IDAES
  download; executables on ``PATH`` still come first.
- Couenne no longer reports small stage-wise models infeasible when a
  feasible network exists. Couenne now defaults to ``feasibility_bt no``,
  ``aggressive_fbbt no`` and ``optimality_bt no`` (set any to ``yes`` in
  ``HENS_SOLVER_OPTIONS_TDM`` to restore it), and Chen's mean temperature
  difference smooths each heater's and cooler's terminal difference rather
  than the cubed base, whose square reached about 1e12 K^6. On the
  segmented-stream test case Couenne now proves the optimum in about 2 s,
  where it previously reached no proof within 60 s. Each setting with only
  some of these off failed one test case on Linux or macOS.
- Segmented exchanger area no longer reports a sliver of the next segment
  when a solved duty passes a segment boundary by solver noise (up to
  ``DUTY_RTOL`` times the duty); the sliver is dropped.
- With the default ``highs`` pinch design solver, a nonlinear pinch design
  model handed to Couenne now takes the missing-Couenne fallback to network
  evolution when Couenne is not installed, as a configured ``couenne`` did.
- The MILP pinch design method no longer rounds below-pinch stage
  temperatures and cooler duties to five decimals when it joins the two pinch
  sides (issue 332). On a stream with a near-flat segment, such as boiling
  over 0.01 K, that rounding moved a stage's heat by several watts and the
  segment area check then rejected the network.
- A Couenne or Ipopt solve that stops early, for example at an iteration or
  time limit, now fails unless the solver returned a point. Before, such a
  solve passed with the variables at their start values, and the unsolved
  network could be post-processed and ranked.
- Synthesis now searches in two directions instead of by quality tier.
  ``HENS_SEARCH_WIDTH`` (0 to 3, default 1) picks the dt_cont multipliers:
  1.0; 0.75, 1.0 and 1.5; five values from 0.5 to 2.0; or seven.
  ``HENS_SEARCH_DEPTH`` (0 to 4, default 1) picks the stages and their effort:
  0 is the heuristic pinch design method alone; 1 adds network evolution; 2
  adds the MILP pinch design method and the thermal derivative method; 3 and
  4 make more heuristic networks, seed evolution with every MILP network and
  evolve more widely. An option set explicitly (``HENS_DT_CONT_MULTIPLIERS``,
  ``HENS_HEURISTIC_PDM_NETWORKS``, ``HENS_PDM_MILP``,
  ``HENS_EVM_N_AD_BRANCHES``, ``HENS_EVM_N_RM_BRANCHES``) overrides its
  level, and those options now default to unset. ``design.heat_exchanger_network()``
  takes ``width`` and ``depth``; a level given in the call also overrides the
  stored options it governs, unless the call sets them. The default run (width
  1, depth 1) sweeps three multipliers with the heuristic pinch design method
  and evolution, where tier 0 ran one multiplier with the MILP.
- Removed: ``HENS_SYNTHESIS_QUALITY_TIER``,
  ``design.enhanced_heat_exchanger_network()``, ``design.open_hens()`` and the
  tier pathways (protected and expanded routes, compact and raw pinch design
  modes). The manifest records ``search_width`` and ``search_depth`` in place
  of ``synthesis_quality_tier`` and the ``selected_pathway_*``,
  ``selected_pdm_mode``, ``selected_tier_origin`` and
  ``selected_protected_pathway`` fields. The Excel input template lists the
  new options.
- The pinch design step uses a heuristic pinch design method by default, in
  ``problem.design.pinch_design()`` and in the main workflow (networks
  labelled ``heuristic_pinch_design_method``). It proposes networks with the
  Linnhoff pinch design rules: pinch matches by the CP and number-of-streams
  rules, stream splits where they fail (as many branches as the partner CPs
  need), tick-off, then matches away from the pinch, the stream nearest the
  pinch first, and utilities on every partner stream. It enumerates the
  alternatives at each decision up to a cap, streams of similar CP first
  (smallest ``|ln(CP_hot / CP_cold)|``). The pinch design side models realise
  each proposal with only its matches allowed, solved with HiGHS when
  highspy is installed whatever ``HENS_SOLVER_PDM`` names, so duties,
  segmented streams, several utilities and multiperiod cases are handled as
  in the MILP; a multiperiod proposal is the union of one design per period.
  Split branches mix at one temperature in the side model, so a split that
  cannot keep its approach that way is also allowed the next stage out.
  Proposals that cannot be solved or verified are dropped.
  ``HENS_HEURISTIC_PDM_NETWORKS`` (10 at depths 0 to 2) caps the networks per
  dt_cont multiplier. Heuristic networks seed network evolution directly,
  never the thermal derivative method. ``method="heuristic_pinch_design_method"``
  is not a design method of its own: the service raises ``ValueError``.
- The MILP pinch design method runs at every multiplier from depth 2 or with
  ``HENS_PDM_MILP``, alone with ``HENS_HEURISTIC_PDM_NETWORKS`` at 0, and at
  any depth in the heuristic's place at a multiplier where the heuristic
  realises no network. From depth 2 its main networks go through the thermal
  derivative method; otherwise its networks seed evolution directly.
- A MILP pinch design side's first, minimum-utility solve stops after 600 s
  unless ``HENS_SOLVER_OPTIONS_PDM`` sets a limit (a limit set there also
  replaces the 60 s fewest-units and minimum-unit limits). With HiGHS its best
  network is then kept as an unconverged result (it seeds later steps but is
  left out of the ranking), and no extra minimum-unit networks are built on
  it.
- Each MILP pinch design side is re-solved for the fewest units (matches,
  heaters and coolers) among those its minimum-utility solve used, at the same
  utility. Minimum utility cost leaves the match binaries free once the targets
  are met, so a solver can keep many matches the network does not need (on the
  pulp mill, thousands of below-pinch matches under 1 kW). The re-solve starts
  from the first solution, stops after 60 s with the best network found, and
  replaces the side only if it converged (or is a HiGHS solve with an
  incumbent) with no more units or utility.
- The MILP pinch design method runs in the heuristic's place at a multiplier
  where every heuristic network is left out of the result (it stopped before
  converging, or fails the feasibility check), not only where none was
  realised. When no converged network remains because a solve stopped at a
  solver limit, the ``WorkflowContractError`` names the options that raise
  the limit (``HENS_SOLVER_OPTIONS_PDM``, ``_TDM`` or ``_EVM``).
- The MILP pinch design method can return several networks per dt_cont
  multiplier. ``HENS_PDM_MIN_UNIT_NETWORKS`` sets how many extra networks with
  the fewest units at the minimum utility cost to collect, and
  ``HENS_PDM_EXTRA_UNIT_NETWORKS`` how many with one unit more (both 0 by
  default and independent of the depth). No-good cuts exclude each topology
  found, each solve is time-limited, the one-more-unit quota is filled after
  the minimum one and counts from the fewest units a side can have, and an
  extra network drops only its empty
  stages, so it stays at minimum utility. Every extra network is a pinch
  design result of its own and seeds network evolution directly. Collecting
  them needs an integer solver (``highs``, ``couenne``, ``scip`` or
  ``mindtpy``), and a failure while collecting them keeps the main network.
- ``design.thermal_derivative()`` and ``design.network_evolution()`` called
  without ``initial_networks`` seed from the stored design of any HEN method,
  not only from a pinch design (for the thermal derivative method) or a
  thermal derivative design (for network evolution). They seed from the
  selected network and the design's other ranked topologies, up to
  ``HENS_BEST_SOLUTIONS_TO_SAVE`` networks, and drop repeated topologies. A
  network without a recorded multiplier runs at 1.0 when it is configured.
- Heat exchanger network synthesis is keyed on a dt_cont multiplier instead
  of an approach temperature. Each multiplier scales every stream's,
  segment's and utility's effective ΔT contribution, as a zone ``dt_cont``
  multiplier does in targeting, so 1.0 designs at the contributions the pinch
  targets use (the old 14 K default ran at 1.4 times them). A stream whose
  ``dt_cont`` multiplier is locked keeps its own contribution, as in
  targeting. A contribution of 0 runs at 0.1 K times the multiplier (0.1 K
  for a locked stream) instead of half the approach temperature.
  ``HENS_APPROACH_TEMPERATURES`` is removed and is now an unknown option, and
  the design accessor's ``approach_temperatures`` argument is gone. Synthesis
  tasks, outcomes, manifests and networks record ``dt_cont_multiplier``
  (manifests ``dt_cont_multipliers``) and no longer an approach temperature or
  ``solver_dTmin`` (a duty-optimised network records its least recovery
  approach as ``minimum_recovery_approach``). To convert an
  approach-temperature grid, use multiplier = dTmin / (2 x
  ``THERMAL_DT_CONT``), so 14 K at the 5 K default is 1.4. The OpenHENS
  fixtures are converted this way and solve as before; the packaged
  four-stream sample now uses the default multipliers, derivative thresholds
  and pinch design solver.
- Every heat exchanger network model is now a native Pyomo model. ``apopt``
  and ``ipopt-GEKKO`` are no longer accepted solver names, and any name other
  than ``highs``, ``couenne``, ``ipopt-pyomo``, ``scip`` or ``mindtpy`` raises
  ``MissingSynthesisSolverError`` listing them. The APOPT solver options
  (``minlp_*``, ``nlp_maximum_iterations`` and the like) are no longer
  accepted, and the ``synthesis`` and ``full`` extras no longer install
  ``gekko``. Large cases no longer fail in GEKKO's recursive conversion (the
  pulp mill failed with "maximum recursion depth exceeded" before any solve).
  A solve that ends without a solution ("noSolution" or "unbounded") is a
  failed solve, so its start values are never ranked as a network.
- ``highs`` is a new pinch design method solver and the new default for
  ``HENS_SOLVER_PDM`` (it was ``couenne``): the HiGHS MILP solver, run in
  process through ``highspy``, which the ``synthesis`` extra now installs. It
  needs no executable, and HiGHS options (``time_limit``, ``mip_rel_gap``,
  ``mip_abs_gap``, ``threads`` and tolerances) are accepted. HiGHS proves the
  same optimum as Couenne on the pinch design model and is much faster on
  large cases, but it can return a different network among those with the
  same objective. A pinch design model that is not linear is solved with
  Couenne instead, with a logged warning, keeping a time limit. Setting
  ``HENS_SOLVER_PDM`` to ``couenne`` uses Couenne for the MILP pinch design
  method; the closest to the previous default design also needs
  ``HENS_HEURISTIC_PDM_NETWORKS: 0`` and ``HENS_DT_CONT_MULTIPLIERS: [1.4]``,
  and the heuristic method always uses HiGHS when it is installed.
- ``scip`` and ``mindtpy`` are new solver names, intended for the thermal
  derivative method (``HENS_SOLVER_TDM``). ``scip`` is the SCIP global MINLP
  solver through ``pyscipopt``, which the ``synthesis`` extra now installs.
  ``mindtpy`` is Pyomo's MindtPy outer approximation, with HiGHS for the MILP
  masters and Ipopt for the NLP subproblems. SCIP limits (``limits/time``,
  ``limits/gap``) and MindtPy settings (``strategy``, ``iteration_limit``) are
  accepted solver options. The default stays ``couenne``.
- Each pinch design model has exactly one objective. With more than one hot or
  cold utility on a side, two objectives were passed and Couenne and Ipopt
  optimised only the first, utility cost. The utility goals keep that
  behaviour, now stated directly; "heat recovery" now minimises utility cost
  plus the heat left unrecovered, and "min units" utility cost plus the number
  of units.
- The pinch design method switches each recovery, heater and cooler duty on
  with a big-M constraint, ``Q <= Q_max * z``, where it required
  ``Q * (1 - z) = 0``. The model is now linear for unsegmented streams, and its
  relaxation is tighter. A solver can pick a different network among those
  with the same objective: on the four-stream example the best network has
  the same units and TAC in 2 stages instead of 3.
- Stage reduction no longer uses APOPT: the reduced network is solved with
  HiGHS when its model is linear and with Ipopt otherwise, for minimum utility
  cost when the case has a choice of utilities. A missing solver, or a
  reduction that would leave no stage, keeps the unreduced network.
- In network evolution's relaxed model, a heater or cooler switched off by the
  warm start could still carry duty with its approach relaxed by big-M, so on
  Nine-stream Linnhoff-Ahmad every evolved network failed verification. Its
  duty is now bounded by its binary, and after the warm start every allowed
  heater or cooler that can sit idle within its approach is switched back on,
  so evolution can still add a utility the parent lacked. A warning says when
  every evolution task fails, so the result was not evolved.
- In multiperiod evolution, a shared exchanger area at zero (a unit with no
  duty) made the area cost's derivative infinite and stopped Ipopt; a 1e-8 m2
  offset under the cost exponent removes it.
- Network evolution starts each candidate from its parent network's solution,
  then adds or removes the match, where each candidate started from default
  values. The flow-on (alpha) values that rank candidates are solved as one
  sparse linear system (0.08 s instead of 9 s on the four-stream example)
  without storing the zero coefficients of inactive matches, which made the
  pulp mill's factorisation run for over 20 minutes. A system with no unique
  solution in [-1, 1] still falls back to neutral alpha.
- The stage-wise models' Chen LMTD approximation no longer takes the cube root
  of a negative number: a heater or cooler at zero duty leaves its approach
  temperatures free, so the base now passes through a smooth max with zero.
- ``ipopt-pyomo`` now defaults to ``bound_push`` and ``bound_frac`` of 1e-6
  (Ipopt's own default is 1e-2, relative to each bound, which is about 3.5 K at
  350 K and moves a warm start off its solved temperatures). Options set in
  ``HENS_SOLVER_OPTIONS_EVM`` or ``HENS_SOLVER_OPTIONS_TDM`` take precedence.
  Ipopt options now go on its command line rather than into an ``ipopt.opt``
  file, so parallel evolution solves no longer wait on each other.
- Duty optimisation without ``min_approach_temperature`` runs at multiplier
  1.0 when it is configured, as it is at every search width.

0.8.4 (2026-11-12)
------------------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The heat pump exchangers now default to finite approach temperatures:
  ``HPR_DT_IHX`` and ``HPR_DT_CASCADE_HX`` 5 K, and ``HPR_DT_CONT`` 1 K (all
  were 0 K). With 0 K the internal and cascade heat exchangers were free
  exchangers of unbounded area. A brewery search, for example, raised the
  suction gas by 75 K through an internal heat exchanger with a 0.1 K approach.
  Every heat pump and refrigeration target moves; set the three options to 0
  for the previous results.
- Heat pump capital is now priced from each machine's components instead of
  its heating duty. Installed capital is 4.16 times the purchased cost of the
  compressors (on suction volume flow), motors (on shaft power) and plate heat
  exchangers (on area), with the correlations of Ommen et al. (2015). It is
  scaled so that Ommen's 1 MW ammonia example costs the IEA HPT Annex 68 level
  the previous model used. Water vapour compressors (closed water stages and
  MVR) are priced as turbo compressors anchored on Zuhlsdorf et al. (2018).

  - Process exchangers are sized interval by interval against the closest
    part of the sink or source each condenser or evaporator can reach. Heat
    exchanged with ambient air goes through an air cooler or heater. Internal
    and cascade exchangers are sized from their own approaches. A condenser
    section counts as gas or liquid against its own stage's condensing
    temperature, and parallel compressor units each have their own motor.
  - New ``HPR_U_TWO_PHASE`` (2000 W/m²K), ``HPR_U_LIQUID`` (1000),
    ``HPR_U_GAS`` (400) and ``HPR_U_AIR`` (50) set the heat transfer
    coefficients.
  - The ``COSTING_HPR_EQUIPMENT_COST``, ``COSTING_HPR_INSTALLATION_FACTOR``,
    ``COSTING_HPR_COST_EXP``, ``COSTING_HPR_COST_FIXED_SHARE``,
    ``COSTING_HPR_COST_STAGE_SHARE``, ``COSTING_HPR_COST_TEMP_FACTOR`` and
    ``COSTING_HPR_COST_TEMP_BASE`` options are removed.
  - Capital now grows with lift, low suction pressure and small approach
    temperatures. Water below about 80 °C is far dearer than before: its
    suction volume is large. Every heat pump target with capital recovery on
    moves.

- TESPy single-stage targets and performance maps now model an internal
  heat exchanger. The search varies it as for CoolProp, held to the
  ``HPR_DT_IHX`` approach; with no gas-side rise the network has none. TESPy
  design points match the CoolProp cycle's COP to 1e-4 with and without it.
- Performance maps keep the target's ``HPR_DT_IHX`` approach at every point.
  Target simulation records carry it as ``internal_hx_approach_temperature``.
  At points with less lift than the target, the CoolProp map simulator ran
  the internal heat exchanger with no approach, and the TESPy one could set
  a suction gas hotter than the liquid heating it. Both overstated its duty.
- Shared multiperiod designs now keep a minimum lift. Each
  vapour-compression stage lifts at least ``HPR_TURNDOWN`` (new, default
  0.25) of its design lift, its largest lift in any period, in every period.
  A period that needs less lift runs at the floor, condensing higher at a
  worse COP, as a fixed machine at part load would. There is no separate
  turndown penalty. A cascade stage's lift includes the cascade exchanger
  approach it lifts across.
- Search and reporting.

  - Each warm start now seeds one restart; the other restarts start from
    Latin hypercube points across the bounds. Every restart used to start
    from the Carnot warm start, so the restarts converged on one design.
    New ``HPR_SEARCH_SEED`` (default 0) shifts the restart seeds and points.
  - Vapour compression targets report ``hpr_details.effective_topology``: the
    requested layout and the stages the design uses, with their fluids,
    temperatures, pressure ratios, work and duties, and a one-line label such
    as "... 3x2 requested; 1 of 4 stages used". A shared multiperiod design
    counts a stage that works in any period. TESPy targets report theirs
    too, so their alternatives are reported as well.
  - ``hpr_details.alternatives`` lists up to three distinct designs (other
    working stages or fluids) within 5% of the target's objective, with
    their topology and objective gap. Very different machines can cost
    nearly the same. Shared multiperiod targets report them too.
  - ``hpr_summary`` includes the topology label and the alternatives.

- Penalised designs, wrong-side heat and the utility capital credit.

  - A final vapour-compression heat pump design that still carries a
    feasibility penalty is no longer a target. The next candidate is tried,
    as refrigeration designs already were. The cheapest feasible point of
    each design the search evaluated is now a candidate as well, so slightly
    penalised points that rank cheaper can no longer leave a small search
    with no target although it found a feasible design. Shared multiperiod designs, heat
    pump or refrigeration, are now checked the same way, period by period:
    a penalty in a lightly weighted period still rejects the design, and the
    failure reports that period's served and targeted heating.
  - Heat a vapour-compression heat pump's condensers reject to ambient air is
    penalised as heat the process could not use. It was free, so a heat
    pump could deliver heat where no process needed it. Refrigerators still
    reject to air freely.
  - ``COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED`` now defaults to
    ``False``. A heat pump is no longer credited with the capital of the hot
    utility and refrigeration plant it displaces, as a retrofit keeps that
    plant. Set it to ``True`` for a new site.
  - New ``HPR_ENFORCE_HEATING_DUTY`` (default ``False``): a vapour-compression
    heat pump must serve its whole targeted heating duty regardless of
    economics, as a refrigerator must serve its cooling. If no design meets
    it, the target fails with the duty the best design serves.

- CoolProp vapour-compression targets may now run a stage transcritical.
  Condensing was capped 2 K below the fluid's critical temperature. A stage
  condensing above it now takes its high pressure from the critical isochore
  at the condensing temperature. Its gas cooler glides down to that
  temperature less the subcooling, and it is held to the discharge pressure
  limit. The warm start is still seeded subcritical, and TESPy targets stay
  subcritical. A stage with no subcooling now reports exactly zero, where
  round-off could give a tiny negative value that failed the simulation
  record. A compressor efficiency correlation (``HPR_COMPRESSOR_MODEL``) is
  given the critical point as a transcritical stage's discharge dew point,
  where it used to fail the candidate.
- Vapour-compression stages are now held to practical limits.

  - Liquid injection: condensate is injected into the compressor discharge
    until the gas is at most ``HPR_MAX_DISCHARGE_SUPERHEAT`` (new, default
    50 K) above its dew point. The condenser rejects the same heat and the
    compressor does the same work, but the hot gas no longer delivers heat at
    discharge temperatures. These reached 400 to 600 °C for water stages and
    220 to 425 °C for ammonia.
  - Penalties, of the stage's condenser duty times the relative violation:
    a pressure ratio above ``HPR_MAX_PRESSURE_RATIO`` (default 15), a
    discharge pressure above ``HPR_MAX_DISCHARGE_PRESSURE`` (new, default
    60 bar; ``HPR_MAX_DISCHARGE_PRESSURE_CO2``, new, 140 bar for carbon
    dioxide), a suction pressure below ``HPR_SCREEN_MIN_SUCTION_PRESSURE``
    (0.5 bar, previously used only by the refrigerant screen), and a lift
    below ``HPR_MIN_STAGE_LIFT`` (new, default 5 K) on a stage that carries
    flow.
  - A two-phase compressor discharge, as dry fluids such as isopentane give
    when compressed from saturated vapour, is penalised by the gas's shortfall
    from its dew point.
  - The pressure ratio limit was a clamp that raised evaporating temperatures
    when a design was decoded. It is now a penalty, so the search sees a slope
    instead of a plateau. The Carnot warm start is still placed within the
    limit, so a small search starts from an unpenalised design.
  - The Excel input template lists the new options and the new approach
    temperature defaults.
  - MVR stages on process vapour are not held to these limits, and TESPy
    targets keep their own model.

- ``HPR_DT_IHX`` is now the hot-end approach of the internal heat exchanger in
  the CoolProp vapour compression models (single stage, cascade, parallel and
  vapour compression plus MVR): the suction gas leaves it at least this far
  below the liquid entering it. It was previously read only by TESPy targeting,
  which rejected a nonzero value; TESPy targets, which have no internal heat
  exchanger, now accept and ignore it. The working-fluid approach
  ``HPR_DT_CONT`` no longer limits the internal heat exchanger (it took twice
  ``HPR_DT_CONT``).
- Fixed refrigeration targets at a load fraction below 1 that failed with
  "failed to return an optimal result" although designs serving the whole
  target existed. The default refrigeration allowed for the untargeted warmer
  cooling was computed from a separate cascade of that cooling, which put the
  split temperature on the problem table's rounded grid. On a near-isothermal
  profile, such as a utility level, the rounding made the allowance a fraction
  of a watt too small, and every design was rejected as leaving cooling
  unserved. The allowance is now the full cooling's need less the target, on
  the cascade the designs are evaluated on. Seen on the DTU brewery with
  utility placement at 56.4% load.
- ``hpr_refrigerant_screen`` now screens every layout a Carnot target's levels
  allow, for the targeting method that would use it, instead of pairing every
  evaporating level with every condensing level as single-stage cycles. Each
  choice of ``n`` condensing and ``m`` evaporating levels gives a single-stage,
  parallel or cascade layout, following the Carnot target's topology, and a
  cascade heat pump also gives vapour compression plus MVR layouts, with the
  MVR on water. Each stage screens the fluids at its own temperatures, so a
  cascade's upper stage is judged on its own small lift. A fluid set takes one
  feasible fluid per stage, with the critical temperature never rising from
  the hottest stage down. The result has ``topology``, ``is_heat_pumping``,
  ``dt_cascade_hx`` and ``layouts`` in place of ``rows`` and ``practical``;
  each layout has ``method``, ``stages``, ``n_fluid_sets`` and
  ``fluid_sets()``, and ``feasible_layouts`` keeps those with a fluid set.
  Explicit levels now need ``topology`` and take ``is_refrigeration``.
- A single-stage or cascade layout with a stage lifting more than the new
  ``HPR_SCREEN_MAX_STAGE_LIFT`` (default 70 K) is also screened as a cascade
  with that stage split into equal lifts within the limit; the inserted levels
  are the layout's ``added_levels``.
- ``HPR_MVR_FLUIDS`` (``mvr_fluids=`` of ``mvr_heat_pump``) now names the
  fluids of the vapour compression stages of a vapour compression plus MVR
  cascade, one per stage, instead of the MVR stages' fluid. The MVR stages are
  open cycles on process or utility steam and always run on water. The option
  rejects water (``Water``, ``H2O``, ``R718``) and defaults to empty, which
  keeps the vapour compression stages on ``HPR_REFRIGERANTS`` as before. Code
  that set ``mvr_fluids=["Water"]`` should drop it; other values now choose the
  vapour compression fluids. ``VapourCompressionMvrCascade.solve`` no longer
  takes ``mvr_fluid``.
- In the refrigerant screen, deep vacuum suction and over ratio limit now rule a
  fluid out of a stage, while near critical fluids stay feasible.
  ``HPR_SCREEN_CRITICAL_MARGIN`` defaults to 5 K instead of 15 K.
- ``vapour_compression_heat_pump``, ``vapour_compression_refrigeration`` and
  ``mvr_heat_pump`` take a refrigerant screen result, or one of its layouts,
  as ``refrigerants`` and run the target once for every fluid set it proposes
  for that method: single-stage and cascade layouts for a cascade cycle,
  single-stage and parallel layouts for a parallel cycle, and vapour
  compression plus MVR layouts for ``mvr_heat_pump``. Each run takes its condensers, evaporators
  and MVR stages from its layout (a split layout's added levels included),
  unless given, when only layouts with those
  counts run; layouts asking for the same counts and fluids share one run. The
  result is an ``HprFluidSetRunsResult`` with every run, failures included,
  and ``best``, the successful run with the lowest total annualized cost. A
  refrigerant list, or one fluid set from ``fluid_sets()``, still returns the
  target. ``mvr_heat_pump`` now takes ``refrigerants`` for its vapour
  compression stages, and refuses ``mvr_fluids`` alongside a screen, since
  the fluid sets choose those fluids.

0.8.3 (2026-11-11)
------------------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New configuration option ``HPR_ETA_MOTOR`` (default 0.95) is the motor
  efficiency of the vapour compression compressors, as ``HPR_MVR_ETA_MOTOR``
  already is for MVR. The electrical work is the shaft work divided by it, and
  the motor losses leave the cycle, so condenser and evaporator duties are
  unchanged. It applies to single-stage, parallel and cascade cycles, the
  vapour compression stages of vapour compression with MVR, and the TESPy
  backend, and each simulation record stage carries its motor efficiency.
  Vapour compression work and costs rise by about 5% at the default; set
  ``HPR_ETA_MOTOR`` to 1.0 for the previous results, and with a compressor
  model that gives overall efficiency, such as ``brendel``, since that already
  includes the motor.
- ``HprPerformanceMapRequest`` and ``HprPerformanceMap`` have a new
  ``motor_efficiency`` field (default 1.0). At 1.0 a map's ``electric_power``
  is compressor shaft power, as before. Set it, for example to the target's
  ``HPR_ETA_MOTOR``, to map electrical power: ``electric_power`` is the shaft
  power divided by it, the COP uses it, and each point satisfies
  ``q_sink = q_source + motor_efficiency * electric_power``. Maps without the
  field still validate.
- New ``problem.target.hpr_refrigerant_screen()`` screens working fluids as
  single-stage cycles between the evaporating and condensing levels of a
  Carnot heat pump or refrigeration target (``target=...``), or between
  explicit ``T_evap`` and ``T_cond``. Each fluid gets its saturation
  pressures, pressure ratio, suction volumetric heating capacity and a verdict:
  above critical, below triple point, over ratio limit, deep vacuum suction,
  near critical or practical. Practical fluids are ranked by suction
  volumetric heating capacity. The candidates default to every CoolProp fluid;
  ``refrigerants="natural"`` selects water, ammonia, CO2 and the
  hydrocarbons. The result is a frozen ``HprRefrigerantScreenResult`` and the
  problem is not changed. The Carnot tutorial notebook shows it.
- New configuration options ``HPR_SCREEN_MIN_SUCTION_PRESSURE`` (0.5 bar) and
  ``HPR_SCREEN_CRITICAL_MARGIN`` (15 K) set the screen's deep vacuum and
  near-critical limits.
- New ``problem.target.hpr_load_kinks()`` finds the logical heat pump and
  refrigeration loads from kinks in the net load profiles. It lists every
  vertex of the sink profile (or, with ``is_refrigeration=True``, the source
  profile) with its load fraction, duty, shifted temperature, slopes and
  ``kind`` (``step``, ``steepens``, ``flattens`` or ``minor``), and the logical
  load fractions: each step, each vertex where the profile steepens by at least
  ``steepening`` (default 2.0), and 1.0. Pass a fraction as ``load_fraction``
  to any HPR target. The method changes nothing in the problem, and has
  ``all_periods`` and workspace mirrors. It returns the new frozen
  ``HprLoadKinksResult`` contract.
- Heat pump and refrigeration targets now accept ``R1233zd(E)`` as a
  refrigerant. Refrigerant names are still upper-cased, so lower-case input
  such as ``r134a`` resolves, but a name is kept as given when CoolProp rejects
  its upper-cased form. ``R1233zd(E)`` was the only CoolProp 8.0.0 fluid
  affected; it raised ``HPRTargetingError`` in preflight.
- Vapour compression and MVR compressors can follow a compressor efficiency
  model instead of a constant isentropic efficiency. ``HPR_COMPRESSOR_MODEL``
  and ``HPR_MVR_COMPRESSOR_MODEL`` take ``"constant"`` (the default, so results
  are unchanged), ``"pareto_front"``, ``"wang_cleland"`` or ``"brendel"``,
  correlations from HeatPumpLab. From Python, the ``compressor_model=``
  argument of ``vapour_compression_heat_pump()``,
  ``vapour_compression_refrigeration()`` and ``mvr_heat_pump()``, and
  ``mvr_compressor_model=`` of ``mvr_heat_pump()``, also take a function of the
  compressor conditions (pressures, suction state, dew points and vapour
  densities), such as a fit of a manufacturer's compressor map. Provenance
  records a function by name. The CoolProp backend applies the models; TESPy
  targeting and performance maps take only a constant efficiency.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Heat exchanger network synthesis, such as
  ``design.heat_exchanger_network()``, no longer crashes with
  ``BrokenProcessPool`` when run from a script with no
  ``if __name__ == "__main__":`` guard on macOS, Windows, or Linux on Python
  3.14. Worker processes there import the main script again, so each one
  re-ran the whole script. Synthesis now logs a warning and solves its tasks
  serially in that case, as the optimisation backends already did. Put the
  script's entry code under a main guard to use worker processes.

Documentation
~~~~~~~~~~~~~

- New Configuration Options page in the API reference lists every option with
  its type, default, unit, limits and description, generated from
  ``configuration_options()``. It includes options such as
  ``HPR_AMBIENT_AIR_ENABLED`` that were only described in the release notes.
- The heat pump guide and methods page now explain ``HPR_AMBIENT_AIR_ENABLED``
  and ``HPR_DT_ENV_CONT``, and how cascade stages pass heat and subcool since
  the 0.8.0 cascade fixes. The HEN synthesis guide now gives each design
  method's default quality tier and explains that unconverged solves
  (``[Unconverged]`` warnings) are left out of the final ranking. The
  cogeneration guide now gives ``isentropic_cogeneration``'s required
  ``efficiency`` and says where band warnings appear.
- Fixed examples that raised: the inverse ``dt_min`` examples on the
  ``PinchProblem`` page named a zone and periods that no sample case has, and
  the zonal guide asked a single-period case for its winter period.
- The ``PinchProblem`` interaction table now lists ``compare_to``,
  ``update_options``, ``set_dt_cont_multiplier``, ``residual_utility`` and
  ``with_utilities_from``, and notes that the ORC methods return ``None``
  without a surplus below the pinch. The ``PinchWorkspace`` table notes that
  ``load`` returns ``None`` for input that does not validate.
- The capability matrix, stability levels and overview lists now include the
  Organic Rankine Cycle, residual utility, area and cost, and utility
  placement methods. The cogeneration methods page lists all five turbine
  methods and says which target's hot utilities they use. The heat pump
  methods page states that cascade topology is the default and which methods
  take the utility-placement flags. The graphs page lists every
  ``problem.plot`` method.
- Fixed examples: the schema-first validation example used ``...``
  placeholders, the HEN serialized-network example used undefined payloads,
  and the utility placement examples now use a small search budget that runs
  in about a second per call.
- The guides now say which extra Excel input needs, that
  ``show_dashboard()`` renders only under ``streamlit run``, that notebooks
  need a Jupyter front end, and that ``openpinch notebook --force``
  overwrites existing copies.
- Developer pages corrected against CI: lanes and the dependency audit, where
  ``OPENPINCH_TEST_REAL_POOLS`` is set, the development install command, and
  the application layer's allowed imports. The two stub reference pages and
  the completed October 2026 audit plan are no longer published, and the
  module appendix is renamed Contributor Module Reference.
- Reorganised navigation. The API Reference lists the public workflow pages
  and the contributor reference pages under separate headings, and the
  Generated Appendix page is gone. Each class and module is now rendered on
  one page only; other pages link to it. The CLI guide moves to
  ``guides/cli.html`` and the Notebooks and Sample Cases guide is folded into
  the Notebook Series page. The old addresses redirect to the new pages.

0.8.2 (2026-11-10)
------------------

New features
~~~~~~~~~~~~

- New ``HPR_AMBIENT_AIR_ENABLED`` option (default ``True``). Set it to ``False`` to
  stop heat pump and refrigeration targets from using ambient air as a free heat
  source or sink. No air is placed, the Carnot search offers no air availability,
  results report zero ambient duty and add no ``AIR`` streams, and cooling water
  still applies.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``design.pinch_design()`` and the synthesis routes that start from it, such as
  ``design.heat_exchanger_network()``, no longer fail when the stream that sets
  the pinch has a supply or target temperature with many decimal places (#297).
  The pinch is rounded to 1e-6 K, so such a stream was kept as a sliver on the far
  side of the pinch, and Pyomo could not build its heat balance. With a cold stream
  from 64.123456789 to 90 degC that sets the pinch, the pinch design used to raise
  ``WorkflowContractError`` and now returns a network. Streams that reach more
  than 1e-6 K past the pinch are split as before.
- New option ``HENS_MIN_EXCHANGER_APPROACH_SHARE`` (#298). The thermal
  derivative and network evolution steps of heat exchanger network synthesis
  choose exchanger approaches by cost, down to 0.1 K, so exchangers could close
  far below the requested ``approach_temperatures``, which only applies to the
  pinch design step. The new option sets a floor as a share of the requested
  approach: at 0.5, a 10 K design keeps every exchanger at 5 K or more. It
  defaults to 0, which keeps the 0.1 K floor and today's results, and must be
  below 1. The HEN synthesis guide now explains how approaches are chosen.
- Network evolution in ``design.heat_exchanger_network()`` no longer returns the
  unevolved network when its starting solve did not converge (#299). IPOPT
  stopping at its iteration limit was reported as a success, and the starting
  model's total annual cost, which can be far below that of any real network,
  was the benchmark every candidate had to beat. Evolution now adopts its best
  verified, converged candidate in that case. A design step whose solve did not
  converge is left out of the final ranking, though later steps still start
  from it, so its objective can no longer win on cost. Such solves are logged as
  ``[Unconverged]`` warnings instead of ``[Success]``.

0.8.1 (2026-11-06)
------------------

Behaviour changes
~~~~~~~~~~~~~~~~~

- When an optimisation task raises under the spawn and forkserver start methods,
  the exception now reaches the caller only after tasks already running in worker
  processes have finished. In 0.8.0 those tasks kept running after the call had
  failed, so they could overlap a retry. The fork start method already waited.

0.8.0 (2026-11-05)
------------------

Behaviour changes
~~~~~~~~~~~~~~~~~

- Optimisation searches no longer pay for worker processes they do not need under
  the spawn and forkserver start methods (macOS, Windows, and Linux from Python 3.14),
  where each worker imports OpenPinch first. A batch of tasks now runs in the calling
  process, a pool starts only if the batch is still running after one second, and a
  worker gets a task only once it has finished importing. A two-stream
  ``carnot_orc(stages=2, maximum_iterations=50)`` drops from about 9 s to 0.3 s under
  spawn, as fast as running serially. Searches long enough to use the pool keep their
  speed-up and can take about one second longer per batch. Results do not change, and
  the fork start method (Linux before Python 3.14) behaves as before.
- ``components.add_process_mvr`` and ``solve_direct_gas_mvr_stream`` now raise
  ``DirectGasMVRPhaseError``, a ``ValueError``, instead of returning a result when
  a stage would cross into liquid. The message names the stage, temperatures and
  pressure. Condensing multi-stage MVR now raises: on a saturated vapour source
  (100 to 99.5 degC) ``n_stages=2`` used to compress the liquid from stage 1, up to
  1159 bar at the default lift. One condensing stage still solves. A source with
  a vapour target now raises when its target is below saturation at a stage outlet:
  100 kW of steam from 120 to 100 degC at 101.325 kPa and pressure ratio 1.5 used
  to report 5881 kW released, 94% of it latent heat outside the source duty. A saved case with such
  a component no longer loads, and an option change that causes it fails and keeps
  the earlier state. Other MVR results do not change.
- ``DirectGasMVRPhaseError`` now also covers a stage at or above the critical pressure
  whose target or inlet is below the critical temperature, which used to be accepted:
  steam from 500 to 350 degC at 15 MPa and pressure ratio 2 left the stage at 30 MPa
  as a compressed liquid. The error's ``above_critical_pressure`` is true, its
  ``t_saturation`` is the critical temperature and its message says so.
- Indirect Carnot heat pump and refrigeration targets now treat a source-side utility
  band on the ambient air or cooling water limit as at that limit, as the vapour
  compression targets do. Previously its whole duty could be charged to refrigeration,
  so these targets can change on a utility background.
- ``HENS_EVM_N_AD_BRANCHES`` and ``HENS_EVM_N_RM_BRANCHES`` now apply to the tier 0
  and tier 1 routes when set, including ``design.open_hens()`` and the protected
  routes of tiers 2 to 5. These routes used to ignore the options and run at a
  breadth of 1, although the manifest reported the option's value. A run with
  either option above 1 now searches more of these routes, takes longer and can
  return a different network, and those tasks get new IDs. Runs that do not set the
  options are unchanged.
- ``target.varbanov_cogeneration()`` and ``TurbineModel.VARBANOV`` are removed.
  They are replaced by ``target.varbanov_small_turbine_cogeneration()`` and
  ``target.varbanov_large_turbine_cogeneration()``, with
  ``TurbineModel.VARBANOV_SMALL`` and ``TurbineModel.VARBANOV_LARGE``, also through
  ``all_periods`` and workspace batches. The old method switched coefficients at
  2000 kW inside the model, which also changed the turbine type, so from 7200 to
  7700 kW on one test case its mass balance never settled. Each new model uses one
  size band and does not switch. Work targets change from the old method wherever
  it would have used the other band.
- A cogeneration target re-run with no usable turbine stage, with or without
  ``base_target``, no longer keeps its earlier values. ``work_target``,
  ``turbine_efficiency_target``, ``turbine_converged`` and ``turbine_result`` are
  now ``None`` and ``turbine_diagnostics`` is empty, as for a run from a thermal
  target. Process-component work, if any, still fills ``work_target``. Previously
  the old work stayed on the target and its result row, so it looked valid.
- If a stage's work is outside the chosen band, the target reports the number and
  a message goes to ``turbine_diagnostics``, a ``UserWarning`` and
  ``problem.report().warnings``; ``turbine_converged`` is not affected. For some
  duties both models warn.
- The grid layout's ``stream_temperatures`` now cover every exchanger a match merges,
  so a stream's supply and target temperatures are its hottest and coldest ends.
  Previously only the first parallel exchanger's temperatures counted. Each match's
  own temperatures and the drawn grid do not change.

Input and validation
~~~~~~~~~~~~~~~~~~~~

- ``active: false`` on a process stream given by ``t_supply``, ``t_target`` and
  ``heat_flow`` was ignored, so the stream stayed in the targets. It now applies
  as it already did for streams given by ``segments`` or ``profile``. Inactive
  streams of any form also no longer set the default utility temperatures. Results
  change only for inputs that set ``active: false`` on a process stream.
- A failed ``update_options`` or other input replacement now keeps retained heat
  pump targets valid, as the rest of the problem's state already was.
- ``ZoneType.UO`` is the new name of the unit operation zone type. ``ZoneType.O``
  and ``ZoneType["O"]`` still work and give the same member, but its ``.name``
  is now ``UO``.
- A legacy ``.xls`` workbook is now rejected with a ``ValueError`` that asks
  for a ``.xlsx`` save. It was accepted as a source suffix but failed with
  ``ImportError: Import xlrd failed``, because no install extra provides
  ``xlrd``.
- Opening a workbook without its Excel reader still raises ``ImportError``, but
  the message now names the missing package and the ``dashboard`` and
  ``notebook`` extras that provide it. ``.xls`` content in a renamed file gets a
  message asking for a ``.xlsx`` save.
- New ``PinchProblem.source_validation_report(source)`` returns a
  ``ValidationReport`` for any source the constructor takes, without raising for
  bad input of any format. ``valid`` is true only if the constructor accepts the
  source. ``problem.validation_report()`` could never report invalid input,
  because the constructor raises first; its docstring and the input guide now say
  so. The ``validate`` command uses the new report, so inline input is checked
  after the same normalisation as loading: for example, a stream entry that is
  not a record is now reported as the loader sees it.

- After ``problem.project_name = "New"``, ``update_options`` and
  ``set_dt_cont_multiplier`` no longer put a root zone named by ``zone_tree`` back
  to its old name, also through ``PinchWorkspace``. The root zone,
  ``to_problem_json``, stored case inputs and results keyed by root zone keep the
  new name, so ``zone_name="New/Area"`` is the path to use. Setting
  ``workspace.project_name`` now renames the live cases and drops their cached
  results; before, it reached only cases built afterwards.

- ``PinchWorkspace.load`` with input that fails while the case is prepared, such
  as an unknown zone type in ``zone_tree``, now stores the case and returns
  ``None``, as it does for other bad input. It stored the case, then raised
  ``ValueError`` from ``self.case(name)``, and ``validation_report(name).valid``
  said ``True``. ``workspace.validation_report`` now uses
  ``PinchProblem.source_validation_report``, so it reports that input as
  invalid. Loading valid input prepares it once more for the check.

Documentation
~~~~~~~~~~~~~

- The tutorial coverage manifest is generated and checked against the committed
  file. Plots, design-result views and read-only calls are now marked
  ``cached observation``, ``optional_dependency`` names real extras
  (``notebook``, ``dashboard``, ``synthesis``, ``tespy``), ``interactive`` and ``slow-hpr``
  notebooks are described as routine CI runs, and ``design_result.views`` maps
  to the multiperiod HEN notebook.
- Corrected the documentation against the code, including examples that raised,
  retired option and notebook names, and the ``openpinch schema`` command. The
  page on adding an analysis method is now a followable checklist.
- Each package directory now has a ``README.md`` stating its interface,
  invariants and side effects. These stay out of the wheel, and a test checks
  that what they cite still exists.
- ``openpinch --help`` now says the command also writes the JSON Schema of each
  public contract.

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``add_process_mvr`` on a multiperiod case no longer fails with "Profile heat
  coordinates must be strictly monotonic" when periods share a stage profile
  breakpoint that differs only by rounding. It failed with or without liquid
  injection. Multiperiod structured stream profiles use the same alignment.
  Single-period results do not change.
- **Behaviour change:** Cascade vapour-compression heat pump targets with one
  condenser and several evaporators no longer deliver the condenser's duty again
  from every lower stage. Those stages condense into the evaporator above and
  have no process condenser, yet each one added the same process heat at its own
  low temperature, so the heat pump moved heat it never needed to and counted it
  in the COP. Targets with ``condensers=1, evaporators>1`` change: on
  ``chocolate_factory`` with ammonia at a 0.25 load and a short search, work goes
  from 55.5 to 42.1 kW, COP from 9.75 to 6.42 and cold utility from 409.6 to
  666.8 kW, with hot utility unchanged at 811.9 kW. ``process_mvr``, which found
  no design at that budget, now finds one. Other topologies are unaffected. In
  ``CascadeVapourCompressionCycle.solve``, a scalar or one-element ``Q_heat`` now
  applies to each process condenser, and lower stages deliver none, where it
  applied to every stage. Calling ``solve`` without ``Q_heat`` follows the same
  rule, using its default of 1.
- **Behaviour change:** In heat pump mode, ``CascadeVapourCompressionCycle.solve``
  now applies a scalar or one-element ``Q_cool`` to each process evaporator except
  the coldest, and the coldest stage takes the remaining cooling. It used to apply
  to every stage, which also fixed the coldest stage's cooling, so the energy
  balance did not close: with ``Q_heat=100, Q_cool=10``, an 80 C condenser and
  24 C and 5 C evaporators on ammonia, ``Q_cool`` plus work came to 52.95 against
  100 delivered. A scalar now gives the same result as ``[10, None]``. With one
  evaporator the value has no effect, because that evaporator balances the cycle.
  Heat pump targets do not use this path and are unchanged, and neither is
  refrigeration mode.
- **Behaviour change:** Multi-stage cascade refrigeration now passes heat from
  each stage's condenser up to the evaporator of the warmer stage. It used to
  pass heat down, from the warm stage's condenser into the cold stage's
  evaporator, and rejected nearly everything at the coldest condenser. With
  several process evaporators, heat from the colder stages was dropped from the
  balance altogether. A single evaporator's ``Q_cool`` is no longer applied to
  every stage, and the top stage now takes the remaining heat, so ``Q_heat``
  values for refrigeration run ``[nan, ...]`` rather than ``[..., nan]``.
  Multi-stage refrigeration targets change: with ammonia, 100 kW of cooling at
  -20 C with condensers at 40 C and 10 C now takes 43.9 kW of work, where the old
  model reported 63.1 kW for 200 kW of cooling. Single-stage refrigeration
  targets are unaffected. Subcooling heat on these stages must also fit in the
  stage's own process duty.
- Cascade and vapour-compression plus MVR targets no longer credit a stage's
  subcooling heat to the evaporator above it, which is too warm to take it. That
  gave the stage impossible COPs, up to about 90% of the Carnot limit. Subcooling
  heat must now fit in the stage's own process duty, so a stage that only passes
  heat up does not subcool. Multi-stage targets rise where this applied, by up to
  about 20% in tests.
- Cascade targets now scale each condenser's subcooling variable by the lift to
  that stage's own evaporator, not the coldest in the cascade. Much of the old
  range was only penalised. The internal heat exchanger's limit also now follows
  the subcooling a stage can apply. Multi-stage targets can change as a result.
- **Behaviour change:** ``update_options`` and ``set_dt_cont_multiplier``, on
  ``PinchProblem`` and ``PinchWorkspace``, now keep components added with
  ``components.add_process_mvr``. These calls used to discard them silently, so
  later targets ran without the MVR. Each component is now re-solved on the
  rebuilt input and its handle stays valid, with its active state kept. Remove
  any code that adds a component again after these calls, as the repeat call now
  adds a second MVR or fails because the source stream is inactive. Efficiencies
  passed explicitly stay as given; efficiencies left unset take the new
  ``PROCESS_MVR_ETA_COMP`` and ``PROCESS_MVR_ETA_MOTOR`` values, so changing those
  options can change a kept component's work. If a component cannot be re-solved,
  the call raises and the problem and its components are unchanged.
- Deactivating one of two process components on the same source no longer turns the
  source back on while the other is still active, which counted the stream twice.
- **Behaviour change:** process components now survive ``to_problem_json``, workspace
  bundles and ``PinchWorkspace.scenario``. They used to be lost with no warning, so a
  saved or copied case targeted different utilities from the original. The input has
  a new optional ``process_components`` field holding each ``add_process_mvr`` call
  and its active state, and loading it re-solves the component. A load fails if a
  saved source stream is gone. Files and bundles without components load as before,
  their export does not gain the field, and the bundle ``schema_version`` stays "3".
  A file that holds components cannot be read by an older OpenPinch, which rejects the
  unknown field. Utility placement on a case with components now places utilities for
  the case with the MVR, not without it, so its utility levels change, and the
  returned case keeps the components.
- Vapour-compression targets now limit each compressor's pressure ratio, with
  the new ``HPR_MAX_PRESSURE_RATIO`` option (default 15). A candidate over the
  limit evaporates warmer, or in a cascade raises its lower levels, so targets
  with a large lift in one compressor can change or find no design. Raise the
  option to relax the limit. TESPy targets and the Carnot models are not
  limited.
- A single-period vapour-compression refrigeration target whose best design
  still carries a feasibility penalty, such as unserved cooling, now raises
  ``HPRTargetingError``. It previously returned that design with the penalty in
  its objective. With a small search budget, raise the budget before relaxing a
  limit: the design may exist.
- Single-stage and cascade vapour-compression targets can now subcool the
  condensate down to the coldest evaporating temperature, the same limit as
  parallel machines. The limit was previously the gap to the top of the sink
  profile, about 10 K when the condenser sits at the top of its load. Targets
  can fall, by up to several percent where subcooling was at that limit.
- A utility level that sits on an air or cooling-water limit can now use that
  service, on utility placement only. A level's band, of up to 0.1 K, ended just
  past the limit, so the whole duty was priced as refrigeration. For example, with ``HPR_DT_CONT`` of 5 a 25 C
  level sits on the 15 C air limit. Utility placement targets with such a level
  can fall by about half: one 25% load test case went from 836.6 to 427.3 k$/y.
- A TESPy heat pump target without the optional ``tespy`` package now raises
  an ``ImportError`` that names the ``openpinch[tespy]`` extra, as the docs
  advise handling. It is still an ``HprTargetCompatibilityError`` (a
  ``ValueError``) with code ``dependency_unavailable``, so existing handlers
  and the command layer's error response are unchanged.
- TESPy vapour-compression targets and performance maps now solve a cycle with
  no subcooling or no superheat. CoolProp refused the saturated state as the
  adapter specified it, so a candidate at the zero-subcooling bound always
  failed, and with a small search budget that could be the only candidate
  reached, raising ``HPRTargetingError``.
- TESPy vapour-compression targets now start their search from the Carnot seed,
  as CoolProp targets do, and honour ``HPR_INITIALISE_SIMULATED_CYCLE``. The
  unseeded first restart could stall where the allocated duty is zero, so with
  a small search budget a target such as R245fa on ``basic_pinch.json`` raised
  ``HPRTargetingError``. TESPy target values can change.
- TESPy vapour-compression targets now keep condensing at least 2 K below the
  refrigerant's critical temperature, the same limit as CoolProp targets. A
  TESPy design could previously sit just below the critical point, where
  CoolProp targets cannot go, so its cost was not comparable. TESPy targets for
  a refrigerant near that limit can cost more.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``HeatExchangerNetworkGridDiagram.save`` to an image file without Kaleido now
  raises ``MissingSynthesisDependencyError``, an ``ImportError``, with the
  ``openpinch[synthesis]`` install command. It used to pass on Plotly's own
  error, which names no OpenPinch extra. Other export errors,
  such as a bad file extension, are unchanged.
- ``design.heat_exchanger_network()``, its multiperiod form and
  ``design.enhanced_heat_exchanger_network()`` now run
  ``HENS_SYNTHESIS_QUALITY_TIER`` when the problem's options set it, and they
  ignored it before. Only problems that set the option see a change: without it
  the methods keep tiers 0 and 2, and ``quality_tier=`` still wins.
  ``design.open_hens()`` stays at tier 1.
- ``result.manifest.method_sequence`` now lists the stages that ran. It always
  listed all three methods, including ``thermal_derivative_method`` on tier 0,
  which leaves it out, and stages that could not run without Couenne.
- ``result.provenance`` now shows the ``HENS_SYNTHESIS_QUALITY_TIER`` that ran,
  matching the manifest. It showed the configured value, for example 1 on a
  default tier 0 run, or 3 on a ``design.open_hens()`` run. The change does not
  affect ``input_fingerprint``.
- Network evolution tasks with no pathway metadata now use the branch breadth of
  the tier that ran, as the manifest's ``evm_n_ad_branches`` and
  ``evm_n_rm_branches`` report. These are the tasks that run when Couenne is
  missing. They read the tier stored in the options, so a stored tier of 4 gave
  a tier 0 or tier 1 run a breadth of 2 and a stored tier of 1 gave a tier 4 run
  a breadth of 1. Their task IDs change, because a task ID includes its settings.

Cogeneration
~~~~~~~~~~~~

- A cogeneration target now says whether its turbine solve converged. Targets and
  result rows carry ``turbine_converged`` and ``turbine_diagnostics``, a solve
  that did not converge or reports a diagnostic raises a ``UserWarning``,
  ``problem.report().warnings`` lists the messages, and the Excel summary has a
  ``Turbine Converged`` column. A weighted-average row is flagged if any period
  did not converge and names the period of each message. These were previously
  dropped, so the mass balance could fail unseen: the Varbanov correlation steps
  at 2 MW, and a duty that puts the first stage near that step alternates
  between the two branches for all 50 iterations. Numeric targets are unchanged;
  a run that did not converge still reports its last iteration.
- Cogeneration targets and their result rows now carry the turbine's stage
  records as ``turbine_result``, a ``TurbineSolveResult`` in fixed units (kW,
  degC, bar, kJ/kg, kg/s). Each period row has its own record, and a
  weighted-average row has none. The Excel summary is unchanged. A stage's
  ``source_index`` now gives its position in the target's ``hot_utilities``; it
  was a position in the list left after skipping unusable utilities.

Streams
~~~~~~~

- Inverting a utility copy no longer detaches the original's segments. For a
  segmented utility, ``get_hot_streams`` and ``get_cold_streams`` with
  ``invert_utility=True`` left the original's segments without a parent
  (``segment.parent`` was ``None``).

Values
~~~~~~

- Building and copying ``Value``, ``Stream`` and ``StreamCollection`` objects costs
  less. A ``Value`` given in its target unit now builds at most two Pint quantities
  rather than four, and copies share read-only magnitudes, so ``deepcopy`` of a
  ``Stream`` took about 14 us rather than 198 us. Four test modules heavy in targeting
  and optimisation used 38% less CPU (183 s to 113 s) in serial runs. Results do
  not change.

- A direct ``Value`` call with a ``period_ids`` key (for example
  ``Value({"values": [10, 20], "period_ids": ["b", "a"]})``) now raises
  ``ValueError``, with the same guidance as a domain ``Stream``. ``Value`` holds
  magnitudes in order and has no period identity, so it dropped the ids and
  gave the first period 10 whatever its id. Give the values in the problem's
  period order. A utility's ``maximum_heat_flow`` and utility placement
  ``maximum_duties``, which key values by period id on purpose, still take it.

Exergy
~~~~~~

- ``problem.target.exergy()`` no longer raises ``AttributeError`` after a heat
  pump or refrigeration target, with or without ``base_target``. The check that
  a selected target has not been modified assumed every graph was a table, but
  exergy adds its curves as plain mappings. Those derived curves are now left
  out of that check, and the enriched target can be selected as a base target.

Reporting
~~~~~~~~~

- ``export_stream_collection`` writes to the working directory when no
  ``output_dir`` is given, as ``export_problem_table`` does. It previously
  wrote to a ``results`` folder beside the package, which is inside
  ``site-packages`` for an installed copy.

Plots
~~~~~

- The hover label of a merged exchanger match in the network grid diagram
  shows the summed area of its exchangers. It previously showed the area of
  the first exchanger only.

0.7.6 (2026-11-02)
------------------

Heat pumps and refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Vapour-compression targets with a refrigerant mixture (for example
  ``R134a[0.5]&R245fa[0.5]``) run about three times faster. The condenser and
  evaporator profiles of a mixture now step in quality inside the two-phase
  region and in temperature outside it, instead of flashing CoolProp at each
  enthalpy, which costs tens of milliseconds for a mixture. Each profile point
  still lies exactly on its isobar, but the points are placed differently, so
  mixture results can move slightly. A fluid's critical point, a costly solve
  for a mixture, is now computed once per process instead of for every
  candidate cycle (new ``OpenPinch.domain.fluids.critical_point``). Pure-fluid
  profiles and results are unchanged. A mixture candidate still costs about
  60 times a pure-fluid one.
- A cascade vapour-compression target no longer charges capital for a stage
  that does no work. The machine cost's stage factor counted every configured
  stage (``n_cond + n_evap - 1``), so a 3x2 cascade whose levels merged still
  paid for four compressors, and could never cost less than a 2x1 or single
  stage design it contains. A stage now counts only when its work exceeds
  0.1 % of the machine's work. Cascade targets whose optimum merges or idles
  stages get a lower TAC and may settle on a different design; single-stage,
  parallel and VC+MVR targets are unchanged.
- Vapour-compression targeting stops each evaporator at its refrigerant's
  minimum temperature, and a cycle asked to evaporate below it raises an
  error. CoolProp extrapolates saturation below that limit without error, so
  targets could run on states that do not exist. A water single-stage heat
  pump on the brewery case with utility placement used to evaporate at
  -0.06 degC, below the triple point of water, and so reach a 265 kW load;
  at 25% load the target now evaporates at 10 degC and its total annualised
  cost rises from about 222,000 to 282,000 $/y. Targets that evaporated water
  below 0.01 degC will change.

0.7.5 (2026-10-27)
------------------

Sample cases
~~~~~~~~~~~~

- The pulp mill sample names its accumulator cooling stream
  ``Cooling od Acc-0`` instead of ``Cooling od Acc.0``. Loading the sample no
  longer warns that the name was changed (dots in names become dashes); the
  stream and its results are the same.

0.7.4 (2026-10-26)
------------------

Command layer
~~~~~~~~~~~~~

- The ``default_utility`` warning says why the default was added. When the
  problem gives utilities of that kind but none is cold enough (or hot
  enough) and uncapped, it reads "None of the given cold utilities is cold
  enough and uncapped; added the default 'CU' as a backup." instead of "No
  cold utility given", and its ``details`` list the ``given`` utilities. The
  dairy chilling case, which gives Cooling Water and Glycol, used to be told
  it gave none.

- A default utility added beside a given utility of the same name (a given
  ``CU`` that is too warm, say) is named ``CU_1``, and only it is treated as
  the backup. Before, the default was renamed only when it joined the
  utilities, and marking went by the name ``CU``, so the given ``CU`` was
  taken for the backup and the added one for a given utility. The default
  is now marked by identity, so this holds whatever names the utilities end
  up with (two given utilities both named ``CU``, for example).

- ``design.optimise_duties`` with a bare heater or cooler name, on a side with
  several utilities, returns an error envelope (``UtilityChoiceError``, with
  ``details`` naming the ``side`` and its ``utilities``) instead of raising
  out of ``run_command``, which engines reported as an internal error. Called
  from Python, it still raises: ``UtilityChoiceError`` is a ``ValueError``,
  and its message now lists the utilities to choose from.

Citation
~~~~~~~~

- Zenodo archived 0.7.3, the first release since the repository was switched
  on: 10.5281/zenodo.23619780 for that version, and the concept DOI
  10.5281/zenodo.23619779 for all versions. The README carries the DOI badge
  and its citation section, and ``CITATION.cff`` names the concept DOI.

0.7.3 (2026-10-24)
------------------

Citation
~~~~~~~~

- The repository has a ``CITATION.cff``. GitHub's "Cite this repository"
  offers the foundational article, and Zenodo reads the file to archive each
  GitHub release with its own DOI, crediting Timothy Gordon Walmsley (ORCID
  0000-0002-9192-0771).

Testing and CI
~~~~~~~~~~~~~~

- When a Pyodide test run fails outside pytest (for example while micropip
  installs the test dependencies), ``scripts/pyodide/run_tests.mjs`` prints
  the Python traceback and exits with status 1. Before, Node printed a
  1.2 MB line of Pyodide's minified source first; GitHub cut it at 64 KB and
  dropped the traceback, so these failures showed no cause.

- The lock file pins pillow 13.0.0 and certifi 2026.10.23, which fix the
  advisories that failed the Dependency audit job against pillow 12.3.0 and
  certifi 2026.6.17. Pillow 13 needs streamlit 1.66.1 (the ``dashboard``
  extra), which drops its ``httptools`` and ``toml`` dependencies.

- CI runs the Pyodide runner's own regression test (a failed run prints its
  traceback) in the Pyodide job, where its Node packages are installed. It
  used to skip in every lane, so the behaviour had no CI coverage.

- CI smoke-tests the ``full`` extra on Ubuntu, so every published extra now has
  an installation check, and a test requires the smoke matrix to match the extras
  in ``pyproject.toml``. Nine fast tests that need GEKKO or Kaleido now carry the
  ``synthesis`` marker.

- The nine-stream live solver benchmark, which pytest always skipped, is now
  ``scripts/benchmark_nine_stream.py``, run by hand. It can take hours and has
  no CI use.

0.7.2 (2026-10-20)
------------------

HEN synthesis orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- In PDM and TDM networks (isothermal mixing), a stream split across several
  matches in one stage reports each branch leaving at the stage's mixing
  temperature, with its split fraction (its share of the stage duty). Each
  branch used to be reported as if it carried the whole stream, so its
  outlet temperature disagreed with its approach temperatures and area: one
  ``crude_preheat_train`` exchanger reported 43.0 m2 of area while its own
  temperatures gave 21.1 m2. Areas are unchanged; the temperatures now match
  them.

- Multiperiod stage reduction after PDM and TDM keeps a stage, a match and
  a heater or cooler that carries duty in any period, and starts the reduced
  model with each of them switched on. It read only the first period, so a stage
  used only in later periods was dropped and the reduced network was solved
  without it, or the reduction failed and fell back to the unreduced model.
  A reduced network that fails the model's own solution checks is not used:
  the unreduced network is kept.

- ``design.heat_exchanger_network()`` no longer takes ``initial_networks``.
  The default workflow never used seed networks and raised for any value;
  ``design.thermal_derivative()`` and ``design.network_evolution()`` take
  them.

- The default ``design.heat_exchanger_network()`` (quality tier 0) runs one
  set of tasks for every value in ``HENS_APPROACH_TEMPERATURES``, as the
  option's help says. It ran only the first value. The result manifest's
  ``approach_temperatures`` lists the approach temperatures the tasks ran,
  including the multiples that expanded tiers explore; it listed the
  configured values whether they ran or not.

- The accepted network is the feasible network with the lowest total annual
  cost from any method stage, and the result's ``method`` records the stage it
  came from. It always came from the last stage that succeeded, so a cheaper
  PDM or TDM network lost to any network evolution result, and the ranked
  networks (``design.top()``, ``design.network(rank=...)``) listed only that
  stage's networks; they now rank every stage's feasible networks by cost.

- Piecewise linearisation of a temperature-heat curve refines its break
  points against the curve in increasing heat. It flipped both arrays
  instead, so for curves listed in increasing heat (cold streams, segmented
  stream profiles, the vapour-compression and MVR evaporator curves) it
  interpolated against a decreasing heat axis: the reported error was
  meaningless (40 K where the fit was 0.16 K off) and the refinement's
  integrity bound was never enforced (first-audit H8). The result no longer
  depends on point order.

- A single-stage network's post-optimisation flow-on factors no longer take
  cold flow-on from the stage itself. Its cold gamma is 0, as for the first
  stage of any network (it was 0.56 where 0 was due on the first audit's
  case), so network evolution ranks single-stage candidates by the right
  benefit. Reported results did not use these factors.

HEN synthesis with several utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Heat exchanger network synthesis uses every hot and cold utility. It used
  only the first of each: ``dairy_chilling_multiperiod`` put every chilling
  cooler on cooling water (20 to 25 C) for streams ending at 2 to 5 C and
  failed, and where the first utility was feasible but another was cheaper
  the result was silently dearer. Each heater and cooler is now one
  (stream, utility) pair with its own duty, approach temperatures, price and
  area, in the stagewise and pinch design models, post-processing,
  verification and the extracted network (one exchanger per pair, named by
  its utility). The pinch design method prefers the cheaper utility when a
  side has a choice. The HEN guide describes multi-utility synthesis.
- The default ``CU`` that input preparation adds as a backup stays out of
  synthesis when real cold utilities exist, as the default ``HU`` already
  did for hot utilities.
- Results with one hot and one cold utility are unchanged.

Utility placement and optimisers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Utility placement with sensible levels no longer fails with
  ``nonpositive_kelvin_region`` when the hottest temperature is more than
  about twice the coldest in kelvin above 0 degC (for example a 320 degC hot
  utility with cooling water, or the Four-stream Yee and Grossmann case for
  every ``sensible > 0``). The sensible span bound is clamped so the lowest
  reachable target stays above absolute zero.
- Placement exergy destruction uses the problem's ``ENV_TEMPERATURE``
  instead of a fixed 298.15 K. The entropy breakdown now has one basis:
  ``utility_entropy`` is the placed utilities' own entropy change and
  ``process_entropy`` the rest of the generation over the process net loads.
  The objective is unchanged. On ``basic_pinch`` (2 isothermal and 1
  sensible level, default options): ``utility_entropy`` 0.5882 to
  1.1230 kW/K, ``process_entropy`` -0.3877 to -0.9224 kW/K, ambient
  298.15 to 288.15 K, ``exergy_destruction`` 59.79 to 57.78 kW;
  ``total_entropy_generation`` stays 0.2005 kW/K.
- The optimised placement case keeps the price and htc of the original
  utility with the same name on the same side, else of the nearest one on
  that side by supply temperature (it had no price and htc 1.0). A hot and a
  cold utility that share a name each keep their own properties. In the Direct scope it
  also keeps that utility's ``dt_cont`` and lists the real temperatures
  that ``dt_cont`` outward of the placed (shifted) levels, so the case
  targets the same levels; in the Indirect and Total Site scopes, and when
  that ``dt_cont`` varies by period, the placed levels keep ``dt_cont = 0``. Tutorial 19's process case now shows its
  utilities 2 degC outward with chocolate_factory's utility price.
- A script without an ``if __name__ == "__main__":`` guard no longer reruns
  the whole placement in every spawned worker: the optimiser runs serially
  and logs how to enable workers. Under Pyodide the optimiser skips the
  process pool, and the search pool is capped at the CPU count.
- The scalar optimiser compares a polished candidate with its start on
  feasibility first, then on the objective, so a constrained problem whose
  global search ends in the infeasible region returns the constrained
  optimum instead of ``NoOptimisationCandidatesError``. Every local method
  polishes within the bounds, and ``local_method`` (also in
  ``UtilityPlacementOptions``) is one of SLSQP, L-BFGS-B, TNC, Nelder-Mead,
  Powell, COBYLA, COBYQA and trust-constr.
- Optimiser options are validated before the solve: ``n_runs``,
  ``maxiter``, ``maxfun``, ``seed`` and ``max_minima`` must be integers,
  ``cluster_tol`` a finite non-negative number, and malformed constraints
  raise ``InvalidOptimisationProblemError`` before any backend runs.
- ``exergy_des_min`` is labelled "Min Exergy Rejected" in summaries and
  "Exergy Rejected Min" in the workbook (was "Min Exergy Destruction" and
  "Exergy Des Min"). It is the exergy of the below-pinch surplus rejected to
  cold utility, not exergy destruction. The field name is unchanged.

Area and unit-count targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``target.heat_exchanger_area_and_cost()`` reports lower area, capital cost
  and total cost targets on problems whose composite curves have a
  temperature gap (a step at constant heat flow, for example between the
  process streams and a utility). This is the second change to area targets
  in two releases: 0.7.1 fixed interval widths that made areas 5-22 % low,
  and this release removes a "discontinuity" rule that made them 1-6 % high.
  At each step the rule replaced a slice's end driving force with the next
  slice's, which collapsed that slice's LMTD. Every slice now ends at its
  left limit, and area targets match an independent fine-slice Bath
  integration. Four-stream Yee and Grossmann: 341.08 to 337.20 m2 (capital
  cost 84,161 to 83,580). ``pulp_mill``: 29,689 to 29,373 m2; its capital
  cost still rises, from 13,462,802 to 28,744,945, because its unit count
  also changes (13 to 88, below). ``basic_pinch``, ``crude_preheat_train`` and
  ``chocolate_factory`` have no gap and are unchanged. Utility placement's
  entropy terms use the same driving-force intervals and never read the
  replaced value, so placement results do not change.
- The minimum number of units (``num_units``) is no longer 0 on about a
  third of problems. Rounding residue of a few 1e-6 kW at the ends of the
  balanced composite curves failed an absolute 1e-6 kW zero test, so only the
  pinch counted as a region boundary. The zero test is now relative to the
  problem's duty, and the first and last rows are always boundaries, so the
  count is N = S - 1 per pinch region. Capital and total cost change with it
  where ``COSTING_HX_UNIT_COST`` is non-zero. ``basic_pinch`` and
  ``crude_preheat_train``: 0 to 9 units (capital cost 442,752 to 1,066,247);
  ``chocolate_factory``: 0 to 110; ``pulp_mill``: 13 to 88. Rows that repeat a
  stream name are separate streams in OpenPinch (``F2`` and ``F2_1``), so
  ``basic_pinch`` counts 9 units, not the 7 of its four physical streams.
- The balance check in area targeting is relative to the duty, so area
  targets no longer raise "requires the inputted composite curves to be
  balanced" on MW-scale problems whose rounding residue exceeds 1e-6 kW.
- Pocket removal flattens the row it inserts at a pocket's exit, and the exit
  row below the pinch when no row is inserted. The exit temperature is
  inserted rounded to 6 dp, so the pocket-free Grand Composite Curve kept a
  bump of up to 0.07 kW on steep segments, and a near-coincident temperature
  could leave a whole pocket row unflattened below the pinch. Utilities were
  then assigned more than their targets (``pulp_mill`` by 0.17 kW, area
  29,372.74 to 29,372.94 m2; one reported case 1,433 kW of cooling against a
  1,350 kW target), and area targeting raised "requires the inputted
  composite curves to be balanced" on about one problem in eight, including
  ``zonal_site_multiperiod``.
- Energy, area and unit-count targets scale with the heat flows from x1e-6
  to x1e3 and are unchanged by a uniform temperature shift. Heat-flow
  thresholds in pocket removal, balanced composite curves and area targeting
  were absolute (1e-6 kW), so at x1e-6 a shallow pocket survived, a utility
  was over-assigned and area targeting raised or drifted by up to 0.1 %.
  They now keep 1e-6 kW from 1 kW up and shrink with smaller problems.
- Composite curve end cleaning (plots and area targeting) trims only rounding
  residue at the curve ends. It used ``np.isclose``'s default relative
  tolerance of 1e-5 of the end heat flow, which dropped real end segments on
  MW-scale curves.
- Composite and net load graphs leave out a series with no heat-flow span,
  such as the cold composite of a problem with only hot streams. It used to
  appear as a segment with no data points.
- Carnot heat pump and refrigeration searches cut short by small iteration
  limits (as in tutorial 08) can stop at a different point, because the
  pocket-exit and curve-end fixes change heat flows by about 1e-6 kW. On
  ``basic_pinch`` with one condenser, one evaporator and 20 iterations the
  heat pump work moves from 53.7 to 77.0 kW. With the default limits the
  result is unchanged (21.85 kW, to 3e-6).
- A direct-integration target whose Grand Composite Curve is zero over its
  whole range (no utility needed, every temperature a pinch) reports the top
  and bottom of that band as its hot and cold pinch temperatures. Both were
  None.
- Heat exchanger network area slicing on segmented streams accepts a
  solver duty that overshoots its stream profile by noise relative to the
  duty (1e-6 of the profile) and drops the overshoot. A 2e-6 kW overshoot on
  a 15 MW match raised "Exchanger duty exceeds an available parent segment
  profile".

The "Total cost now" column under "Utility cost and totals" below predates
these area and unit-count changes. With both, the area target's total cost
on the sample cases is: ``basic_pinch`` and ``crude_preheat_train`` 245,896
$/y, ``crude_preheat_train_multiperiod`` 206,290, Four-stream Yee and
Grossmann 151,080, ``chocolate_factory`` 1,639,540, ``process_mvr`` 20,705,
``pulp_mill`` 55,985,020, ``zonal_site`` 71,327,
``dairy_chilling_multiperiod`` 905,252 and ``zonal_site_multiperiod``
66,564 (its area target no longer raises). Multiperiod cases give their
first period.

Multiperiod HEN pinch design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Multiperiod pinch design (PDM) no longer applies approach constraints to a
  stream in a period where it is absent from that side of the pinch. The
  pinch decomposition sets such a stream to 0 K in that period, so the
  constraints made the shared topology infeasible:
  ``crude_preheat_train_multiperiod`` failed at every dTmin and now solves
  at 5, 10 and 20 K. Those periods also get no approach temperature or
  area from the 0 K values.

HEN utility approach temperatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Heat exchanger network synthesis constrains both ends of every heater and
  cooler to their approach temperature, as it already did for segmented
  utilities. Before, a flat utility had no approach constraint: a hot stream
  could be cooled by recovery below the cooling water outlet and finished
  with a crossed cooler. On a case with cooling water at 320 to 325 K, PDM
  and TDM accepted a cooler at 324 to 322 K (hot-end dT -1 K). That case now
  has no feasible network in the stagewise superstructure, which places
  coolers after the last stage, so synthesis rejects it. In the relaxed
  models that stage reduction and network evolution solve, a heater or
  cooler whose fixed end cannot meet its approach carries no duty, and stage
  reduction keeps only the heaters and coolers the solved network uses in
  some period.
- A synthesised exchanger with a crossed or touching end gets no area. Its
  area used to come from a fallback temperature difference (20 m2 for the
  crossed cooler above); verification now reports its duty without area.
- Network verification checks heaters and coolers for a temperature cross
  and against their utility's approach temperature, which synthesised
  networks now record in ``source_metadata``
  (``hot_utility_minimum_approach_by_period`` and
  ``cold_utility_minimum_approach_by_period``). It checked recovery
  exchangers only.
- The stagewise models' solution check compares the utility cost the solver
  reached with one recomputed from the solved duties. It compared the
  recomputed cost with itself, because post-processing had already
  overwritten the solver's value, so it could not fail.
- Published numbers: no tutorial network had a heater or cooler below its
  approach, but the new constraints change the solvers' paths, and stage
  reduction no longer carries unused coolers. Tutorial 15's selected network
  has total annual cost 154,853.86 $/y with 6 units (was 163,556.12 $/y with
  8 units), the best known network for its case. Tutorial 16's
  ``pinch_design()`` network has 190,449.50 $/y at dTmin 14 K (was
  182,103.64 $/y at 20 K). PDM minimises hot utility only, so every network
  at the target is optimal and their costs are not compared; this result
  moves whenever the solver's path does. Tutorial 16's other designs and
  tutorial 17 are unchanged.

Multiperiod fixes
~~~~~~~~~~~~~~~~~

- Weighted-average summaries pair each period's weight with that period's
  results by period id. A utility placement case whose ``period_ids`` were
  given out of order (for example ``("c", "a", "b")``) applied the first
  weight to the first period it ran, so with weights 0.6, 0.3 and 0.1 on a, b
  and c and cold utility duties of 1000, 1600 and 2200 kW the weighted cold
  utility was 1780 kW instead of 1300 kW. A case placed on a subset of the
  periods (``period_ids=("b", "c")``) now summarises, reports and exports,
  weighting over its own periods; before, each raised "Expected at most 2
  period weight(s), got 3." Placement cases run with all periods in their
  canonical order, as the tutorials do, are unchanged.
- Per-period values (``{"values": [...]}``) without ``PROBLEM_PERIOD_IDS``,
  including those inside a record's ``segments`` or ``profile`` points, are
  a validation error that names the field and asks for one id per period.
  Before, the problem silently had one period and every target used only the
  first value; records of different lengths failed later with an
  unrelated "weights for stream ... must align" error.
- ``StreamSegment.set_attr_for_period`` with a scalar changes only the named
  period, as ``Stream.set_attr_for_period`` has since 0.7.1. Before, it
  overwrote the segment's value, and so its parent's total, in every period.
- A single-period call finds that period's retained targets. After
  ``target.direct_heat_integration`` ran for each period in turn,
  ``target.exergy(period_id=...)`` for any but the last raised "could not
  find a compatible existing target"; it now runs from that period's state,
  as ``target.all_periods`` does.
- The weighted-average ``turbine_efficiency_target`` is total turbine work
  over total isentropic work across the periods, not a weighted mean of the
  period efficiencies. A period with no turbine work no longer pulls it down:
  80 % in one period and an idle period at equal weight now summarise to
  80 % instead of 40 %. Per-period values are unchanged, and so is tutorial
  13's weighted value (10 %, the same in every period).
- Domain ``Stream`` objects reject a ``period_ids`` key in period-value
  data. A stream reads its values in the problem's period order and ignored
  the ids, so ``{"values": [10, 20], "period_ids": ["b", "a"]}`` gave period
  ``a`` 10 rather than 20. Inputs that key values by period id on purpose
  (a utility's ``maximum_heat_flow``, utility placement ``maximum_duties``)
  still take it.

Input and export hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``HENS_SOLVER_OPTIONS_PDM``, ``HENS_SOLVER_OPTIONS_TDM`` and
  ``HENS_SOLVER_OPTIONS_EVM`` accept only known IPOPT, Couenne and APOPT
  tolerances, iteration limits and numeric settings, each with a number, a
  boolean or a single word as its value. Any other option name or value is a
  validation issue, and solver option files and option lines are written only
  from validated options.
- Unit text is checked against a bounded grammar before Pint parses it: at
  most 64 characters, products and ratios of unit names and numbers (Pint's
  ``·`` and ``⋅`` included), one power of at most two digits (``^``, ``**``
  or superscript digits) on a unit name or a group, and at most two
  levels of parentheses. Other unit text raises a ``ValueError``.
- ``workspace.batch`` cannot run another ``workspace.batch``, and one request
  runs at most 100 cases; a larger case list is an error and runs nothing.
- Excel exports (``export_excel``, the ``export.excel`` command,
  ``export_problem_table`` and the dashboard's table export) write every text
  cell as a plain string, never as a formula, and text starting with ``=``,
  ``+``, ``-``, ``@``, a tab or a carriage return gets Excel's quote prefix.

Utility cost and totals
~~~~~~~~~~~~~~~~~~~~~~~

These change published numbers. Energy targets do not change.

- Direct Integration targets report their utility cost, the sum of each
  utility's price times its duty. Before, ``utility_cost`` was always 0 on
  Direct Integration and area targets, and the subzone sums and the workbook
  "Utility Cost" column added up those zeros. ``basic_pinch`` now reports
  17.5 $/h (750 kW of hot and 1000 kW of cold utility at 10 $/MWh).
- Total Site utility lists come from the site utility cascade, so they sum to
  the Total Site hot and cold utility targets. Generation at one level now
  meets use at a lower level, nearest level first; use left unmet is listed
  as hot utility and generation left over as cold utility. Before, the lists
  only netted generation against use at the same level, so they could
  contradict the targets: ``zonal_site`` has a 0 kW hot utility target but
  listed 25 kW of hot water. Its Total Site utility cost falls from
  6.68 $/h to 5.18 $/h (``zonal_site_multiperiod``, first period: 5.36 to
  4.76 $/h).
- Total Site targeting accepts segmented utilities. Before, summing the
  subzone utilities raised "'heat_flow' is derived for segmented stream".
  Each segment now enters the site cascade over its own temperatures.
- The area target's ``total_cost`` is the annualised capital cost plus the
  utility cost over ``COSTING_ANNUAL_OP_TIME`` hours. Before, it was the
  annualised capital alone.
- A per-year output unit for a cost (for example ``OUTPUT_UNIT_UTILITY_COST``
  set to ``$/y``) counts a year as ``COSTING_ANNUAL_OP_TIME`` operating
  hours. Before, the conversion used an 8766 h calendar year, so annual
  utility costs read 5.6 % high at the default 8300 h.

Direct Integration utility cost and area-target total cost on the sample
cases (``COSTING_ANNUAL_OP_TIME`` is 8300 h, 1000 h for Yee and Grossmann;
multiperiod cases give their first period):

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Case
     - Utility cost before ($/h)
     - Utility cost now ($/h)
     - Total cost before ($/y)
     - Total cost now ($/y)
   * - ``basic_pinch``
     - 0
     - 17.50
     - 41,793
     - 187,043
   * - ``crude_preheat_train``
     - 0
     - 17.50
     - 41,793
     - 187,043
   * - ``crude_preheat_train_multiperiod``
     - 0
     - 12.69
     - 40,884
     - 146,190
   * - ``Four-stream-Yee-and-Grossmann-1990-1``
     - 0
     - 67.50
     - 84,161
     - 151,661
   * - ``chocolate_factory``
     - 0
     - 134.53
     - 79,784
     - 1,196,358
   * - ``process_mvr``
     - 0
     - 2.00
     - 4,105
     - 20,705
   * - ``pulp_mill``
     - 0
     - 6,418.29
     - 1,270,793
     - 54,542,579
   * - ``zonal_site``
     - 0
     - 5.18
     - 28,331
     - 71,329
   * - ``dairy_chilling_multiperiod``
     - 0
     - 101.67
     - 62,907
     - 906,741
   * - ``zonal_site_multiperiod``
     - 0
     - 4.76
     - (area target raises)
     - (area target raises)

Input, contracts and option metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Workbook and CSV ``Stream Data`` and ``Utility Data`` tables are read by
  their header row, so columns can come in any order. Before, columns were
  taken by position and a header that put ``t_target`` before ``t_supply``
  silently swapped them. The legacy template spellings (``TS``, ``TT``,
  ``ΔH``, ``ΔTcont``, ``Process Zone``, ``Stream``) are recognised; an unknown
  column is ignored with a warning, two columns for one field are an error,
  and a header row with no recognised name falls back to the old fixed order
  with a warning.
- Blank rows, and rows of blank or whitespace-only cells, in workbook and CSV
  tables are skipped. Before, each became a phantom stream ``S1`` that failed
  validation.
- An ordinary stream may give ``heat_capacity_flowrate`` (CP) instead of
  ``heat_flow``: the heat flow is then CP x abs(``t_supply`` - ``t_target``).
  When both are given and disagree by more than 0.01 percent, validation
  reports an error on ``heat_capacity_flowrate``. Before, the field was
  accepted and ignored. Tables accept a ``CP`` column.
- The Options sheet of ``Excel_Version/Data_input_template.xlsx`` lists the
  current option names, grouped, with their defaults. Its old names
  (``DT_CONT``, ``ANNUAL_OP_TIME`` and so on) were all rejected as unknown.
- ``compare_to`` works between problems with different project names. Before,
  it raised ``KeyError`` because it looked for the base case's scope, which
  starts with the project name, in the other problem.
- Configuration metadata (``config_options()`` and the ``options`` command)
  reports numeric list options (``PROBLEM_PERIOD_WEIGHTS``,
  ``HENS_APPROACH_TEMPERATURES`` and others) as ``number_list`` rather than
  ``string_list``, and ``HPR_LOAD_PERIOD_VALUES`` as ``object`` rather than
  ``number``. A new ``numeric_min_exclusive`` flag is true for options that
  must be greater than their ``numeric_min`` (such as ``THERMAL_HTC`` and the
  efficiencies), which reject 0.
- Utility rows without a name are still ignored, now with a warning when they
  hold data. Stream names that become the same name after normalisation
  (``H.1`` and ``H-1`` both become ``H-1``) are an error. A ``TRUE`` cell in a
  numeric column is an error rather than 1. A project name may contain ``/``;
  before, every load with such a name failed.

Testing and CI
~~~~~~~~~~~~~~

- The four-stream warm-start solver regression accepts either of the two
  checked-in OpenHENS solutions it can reach: the best network (154,853.85
  $/y) with the macOS IDAES solver binaries, or the second-quartile one
  (159,038.72 $/y) with CI's Ubuntu 22.04 binaries. It had been changed to
  expect only the best one, which failed the solver lane on the 0.7.2 release
  pull request. It still checks the network's structure, heat balances and
  cost against whichever solution was reached.
- The utility placement timing tests
  (``tests/analysis/utility_placement/test_performance.py``) carry the
  ``performance`` marker, so they run in the performance lane and no longer in
  the unit lanes. Their wall-clock budgets failed on busy shared runners with
  no code change: a bound reduction that takes about 10 ms measured 0.5 s on
  ``unit-py312`` against a 0.45 s budget.

Heat pump second law
~~~~~~~~~~~~~~~~~~~~

- Cascade Carnot targeting (``carnot_heat_pump()`` and
  ``carnot_refrigeration()`` with more than one condenser or evaporator) no
  longer moves heat uphill for free. The heat-engine and recovery branch
  pooled every condenser and evaporator that took part in any downhill pair,
  so heat from a cold evaporator could reach a hotter condenser with no work,
  and the optimiser exploited it. Duty is now allocated pair by pair, from an
  evaporator to a condenser at or below its temperature, and a pair earns work
  only when ``HPR_INTEGRATED_EXPANDER_ENABLED`` is on (it already set the
  heat-engine efficiency to 0 when off; the field help now says so). A Carnot
  cycle is solved only if its entropy generation, sum(Q/T) delivered minus
  absorbed, is non-negative, and targeting rejects an unsolved cycle as an
  infeasible candidate. Published numbers change for the default call
  (three condensers, two evaporators, ``load_fraction=0.5``):

  - ``basic_pinch``: work 0.0005 kW to 62.07 kW, ``hpr_cop`` 4.32 to 6.04,
    hot utility 375 kW in both; entropy generation -0.124 to +0.069 kW/K.
  - ``pulp_mill``: work 0.23 kW to 24,330 kW, ``hpr_cop`` 2.54 to 3.20, hot
    utility 97,115 kW to 77,775 kW; entropy generation -43.7 to +28.8 kW/K.

  One-condenser, one-evaporator and parallel Carnot targets are unchanged,
  so tutorials 08 and 10 keep their numbers. Cascade vapour-compression
  targeting seeds from this optimiser; the tutorial runs (09, and 10's
  3x2 cascade) give the same results.

Heat pump conventions, exergy and targeting details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hpr_cop`` follows the duty for every cycle model: heat pumps report the
  heating COP (Q_cond/W) and refrigeration the cooling COP (Q_evap/W). Carnot
  refrigeration reported the heating COP, so its ``hpr_cop`` falls by 1 for
  the same design: tutorial 08's refrigeration 1.58 to 0.58, tutorial 10's
  Carnot refrigeration 2.38 to 1.38. Vapour-compression, MVR and TESPy targets
  already reported this way and are unchanged.
- The Carnot heating COP stays 1 + eta x T_l/(T_h - T_l): ``HPR_ETA_II_CARNOT``
  applies to the cooling Carnot COP and the heating COP follows from the
  energy balance. The field help and the Heat Pump guide now say so.
- Vapour-compression streams keep their full duty when the evaporator inlet is
  exactly saturated liquid, as in a zero-lift cascade stage. The first profile
  point was dropped, so the stream carried 95 percent of the stage's duty and
  ``hpr_load`` was out of balance (0.55 kW in the audit's ``basic_pinch``
  two-by-two water cascade).
- ``ETE`` is the exergy transfer effectiveness of the Balanced Composite
  Curves: the exergy gained by the cold balanced composite over the exergy
  released by the hot one, relative to ``ENV_TEMPERATURE`` and with utilities
  included; below ambient the roles swap, so refrigeration problems are
  covered. It is at most 100 percent. It was the above-pinch deficit exergy
  over the below-pinch surplus exergy, which is not an effectiveness and was
  458 percent on ``pulp_mill``. The new ``exergy_released`` field is its
  denominator and weights it across periods. Direct Integration values change
  from 100.5 to 77.1 percent on ``basic_pinch``, 458.7 to 69.0 percent on
  ``pulp_mill``, 108.9 to 73.5 percent on ``chocolate_factory`` and 28.8 to
  75.3 percent on ``Four-stream-Yee-and-Grossmann-1990-1``.
- The warning for a negative stream or utility ``dt_cont`` now says that the
  reported heat recovery can then exceed the thermodynamic limit.
- With assisted heat transfer (``DIRECT_ASSISTED_HT_ENABLED``), heating demand
  in a pocket below the pinch is met by hot utility, so the utilities balance
  the targets. On ``zonal_site`` the hot utilities rise from 0 to 44.65 kW (LPS
  17.22 kW, HW 27.43 kW) and the "unmet hot utility" warning is gone.
- Total Site targeting no longer rounds its stored Problem Table to 4 decimal
  places (only the graph copies are rounded), and the energy transfer diagram
  is built from each target's unrounded table. On ``pulp_mill`` the diagram's
  operation cascades moved by up to 8.3 kW.
- The vapour-compression docstrings say that T_evap is the dew-point and T_cond
  the bubble-point temperature, as the code uses them.

0.7.1 (2026-10-19)
------------------

Network grid data
~~~~~~~~~~~~~~~~~

- A network's grid layout (``design.grid_layout()``, and ``view.grid`` in
  ``design.`` command responses) now gives each stream row its supply and
  target temperature in the period (``stream_temperatures``, the ends of its
  exchangers' temperatures) and each match its heat transfer area (summed
  over the exchangers one match merges). A client can draw a full grid
  diagram, with end temperatures and areas, from the layout alone.

Testing and CI
~~~~~~~~~~~~~~

- The Pyodide test lane is required by ``CI OK``, so a change that breaks the
  web app's engine blocks the merge. Its PyPI test dependencies are pinned
  to the ``uv.lock`` versions in ``scripts/pyodide/requirements.txt``, a test
  keeps the two in step, and the run fails if micropip fetches anything
  else from PyPI. ``tests/presentation`` now runs in the lane's core shard,
  and the utility placement timing tests allow WebAssembly ten times their
  native budget.
- ``tests/conftest.py`` loads an ``openpinch`` Hypothesis profile
  (``derandomize=False``, no deadline, no example database), so
  ``--hypothesis-seed`` picks the same examples on GitHub, locally and in
  Pyodide. Before, Hypothesis's own CI profile ignored the seed on GitHub.

Result fixes
~~~~~~~~~~~~

- ``target.heat_exchanger_area_and_cost()`` reports higher area, capital cost
  and total cost targets. Inserting temperatures into a problem table left the
  inserted intervals and their neighbours with the wrong widths and enthalpy
  steps, so the area integration over the balanced composite curves came out
  5-22 % low. For example, the ``basic_pinch`` area target rises from about
  475 m2 to about 554 m2. Published area and cost numbers change; energy
  targets do not.
- ``run_command`` with ``design.multiperiod_heat_exchanger_network`` returns
  its result. Before, building the response's ``view`` raised "period_id is
  required when a network has multiple period states" for the shared
  multiperiod network. ``design.*`` responses now carry ``views``, one
  ``HeatExchangerNetworkView`` per period of the rank-1 network in period
  order (one for a single-period network), and ``view`` is the first of them.
  The HEN design view gains ``views(rank=)``, which returns them.
- ``export_excel()`` (and the ``export.excel`` command) after only
  ``target.all_periods.<method>`` writes that run's period sheets and the
  weighted average, as ``include_periods=True, include_weighted_average=True``
  does. Before, it raised "No target analysis is available". With a
  single-period run, or with either flag given, the workbook is unchanged.

Multiperiod and WebAssembly fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Setting one period's value on a stream with a single (scalar) value in a
  multiperiod problem, for example
  ``stream.set_attr_for_period("heat_flow", 50, period_id="c")``, changes only
  that period. Before, it wrote the value to every period. Assigning a
  segmented utility's duty for one period had the same fault.
- Heat exchanger network synthesis with ``HENS_MAX_PARALLEL`` above 1 runs
  in the browser (Pyodide). It solves each stage's tasks one at a time where
  the runtime cannot start worker processes, instead of raising
  ``NotImplementedError``. The packaged Four-stream sample sets
  ``HENS_MAX_PARALLEL`` to 10.
- The weighted-average summary of a shared-design multiperiod heat pump or
  refrigeration run works when one period has no heat pump load. Before,
  building that period's zero-load row raised a validation error on
  ``hpr_period_ids``, so ``report()`` and the weighted summaries failed. The
  zero-load row keeps the shared design's values (``hpr_design_vector``,
  ``hpr_period_ids``, ``hpr_period_weights``, ``hpr_period_success`` and
  ``hpr_weighted_objective``), so the weighted-average row reports them too.
- A shared-design (``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``) utility heat pump
  or refrigeration target with ``include_subzones=True`` reports the same
  subzone and base rows as a single-period target. Before, preparing the
  shared design left each period's base targets on the zones, so a run for
  one period kept only its heat pump rows. A zone with no heat pump or
  refrigeration load in the selected period now gets no target, as in a
  single-period run, instead of raising "requires a non-zero HPR load". A
  period with no load runs no machine, so it has no case in the shared design,
  but its weight still counts: the weighted operation (duties, work, operating
  cost and penalty) is spread over the whole schedule, while the COP and the
  peak capital are unchanged.

Command layer
~~~~~~~~~~~~~

- ``workspace.set_case_input`` takes ``source`` as ``{"data": ...}`` or the
  TargetInput JSON and rejects anything else, such as a file path, with
  ``ValueError``. Before, a path string read that file on the engine host.
- A unit string pint does not know, such as ``{"value": 200, "unit":
  "bogus"}``, is a validation issue on its field: ``build_validation_report``
  returns ``valid=False`` and the ``validate`` command returns its report.
  Before, both raised ``UndefinedUnitError``.
- A ``zone_tree`` whose zone has two children of the same name is rejected
  when the input is loaded or validated ("Duplicate zone name ...").
  Before, the second child was merged into the first.
- ``Value.to`` converts an offset temperature (``degC``, ``C``, ``°C``,
  ``degF``) to a difference unit such as ``delta_degC`` as a difference, as
  the ``Value`` constructor does, so a ``Stream`` takes ``dt_cont`` as
  ``{"value": 5, "unit": "degC"}``. Before, it raised
  ``DimensionalityError``.
- ``run_command`` raises ``ValueError`` naming the accepted shapes,
  ``{"data": ...}`` or ``{"sample_case": ...}``, when ``problem`` is a bare
  string or names neither. Before, it raised ``AttributeError`` or
  ``KeyError``.

0.5.1 to 0.7.0
--------------

Tutorials
~~~~~~~~~

- Tutorial 15 asks ``design.utility()`` for its case's hot utility by its
  name, ``HPS``. It asked for ``HP Steam``, which used to answer 0 and now
  raises as an unknown utility, so the notebook stopped running.

Result fixes
~~~~~~~~~~~~

- ``export_excel(include_periods=True)`` writes each period's problem tables
  from the all-periods run that produced the exported results. Before, a later
  single-period run replaced that period's tables, so the workbook mixed the
  batch's summary with the later run's tables.
- In ``target.all_periods.<method>`` (and a single-period target on a
  multiperiod problem) each target's utilities now carry that period's duty as
  a scalar. Before, each utility gave an array over all periods in which only
  the target's own entry was set.
- After ``target.all_periods.<method>`` with no single-period run,
  ``plot.data()``, ``plot.catalog()``, the plot methods and the plot exports
  read the graphs of every period's result. The catalogue gains a
  ``Period ID`` column, and the graph data is keyed ``"<graph set> [<period>]"``.
- A single-period ``target.<method>`` run no longer discards the last
  all-periods results, so ``report(include_periods=True)`` and the weighted
  summaries still work after it. Changing the problem's input still clears
  them.
- ``design.top(n)`` uses the same ranking as ``design.network(rank=)`` and
  ``design.grid(rank=)``: successful, feasible outcomes of the accepted method
  with duplicate network structures removed. Before, it sliced
  ``result.ranked_networks``, so rank ``k`` could differ between them.

Fuller results
~~~~~~~~~~~~~~

- Heat pump and refrigeration target rows (``TargetResults``) carry the values
  of ``hpr_summary``: ``hpr_selected_load`` and ``hpr_achieved_load`` (kW),
  ``hpr_objective``, and for a shared multi-period design
  ``hpr_design_vector``, ``hpr_period_ids``, ``hpr_period_weights``,
  ``hpr_period_success`` (by period) and ``hpr_weighted_objective``.

- ``summary_frame()`` and ``report().metrics`` include ``Area``, ``Units``,
  ``Capital Cost``, ``Total Cost``, ``ETE``, ``Exergy Sources``,
  ``Exergy Sinks``, ``Min Exergy Requirement`` and ``Min Exergy Destruction``
  once the area and cost or exergy targets have run. Without those targets
  the columns and metrics are unchanged.

- ``utility_placement_result.retarget_comparison`` lists each placed utility
  level of the winning periods (``UtilityPlacementRetargetRow``: period,
  side, utility, kind, placement rank, supply and target temperatures,
  ``optimizer_duty``, ``retargeted_duty`` and ``is_fallback``), so the
  optimiser's allocation and the placed case's retargeted duty sit side by
  side. Results recorded before have an empty table.

- The HEN design view gains ``view(rank=, period_id=)``, a
  ``HeatExchangerNetworkView`` with the duplicate-free ranking table (rank,
  method, status, objective, total annual cost, unit count and duty totals,
  as ``top(n)`` ranks them), one ranked network's totals and utility duties,
  and its grid layout; and ``grid_layout(rank=, period_id=)``, the grid
  diagram's stream rows, stages, matches (with duties and temperatures) and
  branch counts as data, without plotly. ``run_command`` returns the view as
  ``view`` beside ``value`` for ``design.*`` commands, and the contract is in
  ``contract_json_schemas()``.

Command layer
~~~~~~~~~~~~~

- ``run_command`` reports a session it does not hold (in ``session`` or a
  ``{"$session": ...}`` argument) as an ``error`` with type
  ``unknown_session`` and ``details.session``, instead of raising
  ``ValueError``.
- A heat pump target the engine cannot run, such as a TESPy target without
  the ``tespy`` extra, returns an ``error`` of type
  ``HprTargetCompatibilityError`` with ``details.code`` (for example
  ``dependency_unavailable``) instead of raising.
- Every ``run_command`` response has a top-level ``warnings`` list of the
  non-fatal issues met while the command ran, each ``{type, message,
  details}`` like ``error`` (empty when there are none). Input validation
  warnings come one per issue, with type ``validation`` and the
  ``ValidationIssue`` as ``details``; the library raises them as an
  ``InputValidationWarning`` (a ``UserWarning``) carrying its ``issues``.
  Loading a problem that gives no hot or no cold utility adds a
  ``default_utility`` warning (``details``: ``name`` and ``kind``) for each
  default utility added.
- ``contract_json_schemas()`` defines ``Value`` and ``StreamCollection`` once
  under ``$defs`` and refers to them, so generated client types have one
  ``Value`` and one ``StreamCollection`` type.
- ``PinchWorkspaceBundle`` has an optional ``active_case``:
  ``to_bundle()`` records the active case and ``from_bundle()`` activates it
  (the default case when it is absent). A bundle that names an active case
  it does not hold is rejected. Releases before this one reject bundles that
  carry ``active_case``.
- ``PinchWorkspace.set_case_input(case_name, source)`` and the
  ``workspace.set_case_input`` command replace an existing case's input and
  return its validation report. The session and the other cases' results are
  kept.
- ``ConfigurationFieldMetadata`` gives each field's ``unit``,
  ``description`` and ``default``, and ``label`` is a display name such as
  "Minimum contribution temperature difference" rather than the title-cased
  field name.
- An ``options`` command returns the editable configuration fields
  (``fields``), the case's effective values (``values``) and the options its
  input sets (``set``), so an editor can tell an option set to its default
  from one left unset.
- An ``import`` command reads problem data from file contents (base64), each
  file with its ``role``: one ``problem`` file (JSON or a workbook) or a
  ``streams`` and a ``utilities`` CSV file. It returns the data, its source
  kind and project name without building the problem, so invalid data can
  still be edited. ``load_problem_source_from_files`` is the adapter behind it.
- ``plot.export``, ``plot.export_gallery`` and the ``export.plots`` command
  take ``include_plotlyjs``: ``True`` (embedded, the default), ``"cdn"``, or
  ``"directory"`` for one shared ``plotly.min.js`` beside the files.
- ``plot.catalog()`` (and the ``plot_catalog`` command) has a
  ``Plot Method`` column: the ``problem.plot`` method, and ``export.plots``
  ``plot`` name, that draws each graph (empty for graph types no method
  draws).
- ``design.utility(name)`` takes the problem's utility name (``"HPS"``) as
  well as the network's stream name (``"Hot Utility.HPS"``); before, the
  utility's own name answered 0. A name that is not one of the problem's
  utilities raises ``ValueError`` instead of answering 0, as does a name
  shared by a hot and a cold utility (ask for ``"Hot Utility.<name>"`` or
  ``"Cold Utility.<name>"``).
- ``design.multiperiod_heat_exchanger_network(prerequisites=True)`` runs
  ``target.all_periods.all_heat_integration()`` first when the problem has
  no all-period targets, so a fresh problem (or a session-less command) can
  run it. Without the argument it still asks for the targets first.

Progress reports
~~~~~~~~~~~~~~~~

- Heat pump targeting, utility placement and HEN synthesis report progress
  while they run. ``run_command(request, progress=callback)`` calls
  ``callback`` with each report as JSON-ready data: ``analysis``, ``stage``,
  ``message``, ``done`` of ``total`` ``unit``, ``best_objective`` (and the
  ``objective`` it measures), ``elapsed`` seconds and ``scope`` (the period
  of an all-periods run). ``run_command_json(request, progress)`` passes
  each report as JSON text. Reports come at most every 0.25 s per stage.
- Heat pump targeting reports warm starts, multistart runs finished of the
  multistart count (with the evaluations so far) and the ranked candidates
  checked. Utility placement reports evaluations of the evaluation limit in
  its search and polish, then the candidates replayed. HEN synthesis reports
  tasks solved of each method stage's tasks and the lowest total annual cost
  so far. Work in worker processes is counted when its batch returns.
- An exception the callback raises cancels the command: it propagates from
  ``run_command`` unchanged and leaves the session as it was.
- In Python, ``OpenPinch.domain.progress.reporting(callback)`` receives the
  same reports as ``Progress`` objects for any analysis run inside it.

Example workbooks
~~~~~~~~~~~~~~~~~

- ``examples/OpenPinchWkbs/Bjork and Pettersson.xlsb`` loads: its Utility
  Data sheet no longer has a literal ``%`` in ``dt_cont``. Every bundled
  example workbook now loads.

Engine commands for the tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``run_command`` gains problem commands: ``validate`` (normalised input and
  ``ValidationReport``), ``to_problem_json``, ``streams`` (prepared streams
  and utilities, one row per segment), ``report`` (metrics, ``ProblemReport``
  and the summary table, with period and weighted-average rows),
  ``compare``, ``residual_utility``, ``with_utilities_from`` and
  ``components.add_process_mvr`` / ``inventory`` / ``activate`` /
  ``deactivate`` for process MVR components in a session.
- A request's problem can carry ``project_name``. A second problem is passed
  as ``{"$session": <id>}``.
- A returned case (utility placement) also carries its retargeted
  ``results`` and its ``utility_placement_result`` evidence.
- Heat pump targeting failures (``HPRTargetingError``) come back as
  ``error`` with the ``hpr_failure_summary`` details instead of raising.
- HEN ``thermal_derivative`` and ``network_evolution`` accept seed networks as
  JSON, as returned in a design result: one network or a list of them.
  Synthesised networks now also record the solver approach temperature in
  ``summary_metrics["approach_temperature"]``, so a JSON seed runs its seeded
  tasks at the approach it was solved at.
- Workspace sessions: ``open`` with ``workspace`` (a baseline problem or a
  bundle), ``case`` to pick the case a command runs on, and
  ``workspace.scenario`` / ``use_case`` / ``update_options`` /
  ``set_dt_cont_multiplier`` / ``add`` / ``compare_cases`` / ``to_bundle`` /
  ``batch`` (one command on each case, with per-case errors).
- ``plot_catalog``, and ``export.excel`` / ``export.plots`` returning the
  written files as base64; files go to a private temporary directory.
- New ``PinchWorkspace.to_bundle()`` and ``PinchWorkspace.from_bundle()``
  build and read a workspace bundle in memory; ``save_bundle`` and
  ``load_bundle`` use them.
- Workspace commands (``workspace.add``, ``workspace.to_bundle`` and the
  rest) also work on a workspace with no cases yet, such as one opened from
  an empty bundle.
- Command responses are strict JSON: NaN and infinities come back as
  ``null``.
- ``validate`` on invalid inline input returns ``{"input": null, "report":
  ...}`` with the ``ValidationReport`` issues instead of raising
  ``ValueError``. Other commands still raise on invalid input.
- ``ValidationReport`` gives one issue for a bad value field such as
  ``t_supply``, from the value form its input has (number, ``value`` or
  ``values``), instead of one issue per form.
- ``load`` returns the prepared problem: its ``input`` (a sample case's
  ``TargetInput`` too), its ``streams`` (as the ``streams`` command) and its
  ``zones`` tree, each zone with its address, type, process streams by name
  and subzones.

Contract JSON Schema
~~~~~~~~~~~~~~~~~~~~

- New ``OpenPinch.contracts.schema.contract_json_schemas()`` returns the JSON
  Schema of each public contract (``TargetInput``, ``TargetOutput``,
  ``PinchWorkspaceBundle``, the HPR performance map, utility placement,
  dt_min and network synthesis results), so typed clients such as a
  TypeScript frontend can be generated from them. Inputs use the validation
  schema and results the serialisation schema; ``Value`` is described by its
  JSON form (scalar or per-period values with a unit).
- New ``openpinch schema -o DIR`` writes one ``<Contract>.schema.json`` file
  per contract.
- The stream collection schema (``hpr_hot_streams``, ``hpr_cold_streams``)
  requires every column and rejects unknown keys.

Engine commands
~~~~~~~~~~~~~~~

- New ``OpenPinch.application.commands.run_command`` runs one
  ``target.<method>`` or ``design.<method>`` command on a problem given as
  JSON (``{"data": ...}``) or as a packaged sample case name, and returns the
  method's contract result and the problem's ``TargetOutput`` as JSON-ready
  data. ``run_command_json`` does the same with JSON text, for Pyodide. Both
  the browser engine and the local backend of the OpenPinch web app call it.
  Sample cases are resolved by name only and problem data must be a JSON
  object, so a request cannot make the engine read files from its host.
- Engine sessions: ``"open"`` loads a problem and returns a ``session`` id,
  later requests pass ``session`` instead of ``problem`` to run on the same
  live problem, and ``"close"`` discards it. Commands that need an earlier
  command's state now work, such as ``design.multiperiod_heat_exchanger_network``
  after ``target.all_periods.<method>``.
- Commands may be deeper paths such as ``target.all_periods.<method>``; private
  names are rejected at every level.
- In a session a returned target comes back as ``{"ref": <id>, "target": ...}``
  (the target's results row, which is the only place a ``base_target``
  command's residual target appears), and an
  argument ``{"$ref": <id>}`` passes it to a later command (for example as
  ``base_target``). JSON arguments for parameters typed as a contract, such as
  the ``request`` of ``target.hpr_performance_map``, are validated into it.
- A command that returns a new case, such as ``target.utility_placement``,
  returns the case's problem data as ``{"data": ...}`` instead of nothing; in
  a session the case also opens as a new ``session``.

Heat pump cycle states at saturation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- When a pressure-enthalpy flash fails and the enthalpy is within round-off
  (a quality within 1e-9) of saturated liquid or vapour, on either side, the
  vapour-compression cycle now uses exactly saturated liquid or vapour.
  Before, an enthalpy a few ulps above saturated vapour was treated as
  superheated, which could need a temperature below the triple point and fail
  the candidate, while one a few ulps below gave a quality of 0.99999999999993.
  Native and WebAssembly CoolProp differ in those last digits, so the heat
  pump search took different paths. Short searches can now settle on a
  different optimum: tutorial 09's quick water heat pump moves from COP 7.72
  (24.3 kW) to COP 6.88 (27.2 kW), the same as in the browser.

CMA-ES samples alike on every platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The CMA-ES backend (the HPR and utility placement default) now
  eigendecomposes its covariance by parallel Jacobi rotations (elementwise
  numpy operations, about 16 ms for 50 coordinates) and forms its matrix
  products from correctly rounded dot products (``math.fsum``), instead of
  calling LAPACK ``eigh`` and BLAS.
  LAPACK and BLAS builds differ in the last bits and in eigenvector signs:
  from the second generation on, the browser (Pyodide) and the desktop sampled
  candidates about 1e-16 apart, and a short heat pump search then settled on a
  different optimum. Given the same objective values, CMA-ES now visits the
  same points on macOS, Linux, Windows and Pyodide. Short searches can settle
  on a different optimum than before.
- Composite and grand composite curves drop a point that lies within tol of
  the line through its neighbours in heat flow, not only in temperature. On
  a nearly vertical segment (a duty span of 1e-7 kW over several degrees) the
  temperature residual crossed tol with a one-ulp change in a duty, so the
  browser and desktop kept different points.

Running in the browser (Pyodide)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``workers`` and ``max_parallel`` fall back to serial execution where Python
  cannot start threads (CPython built for Emscripten, as in Pyodide). Before,
  ``all_periods`` targets and inverse dt_min with ``workers > 1``, and branched
  HEN network evolution, failed there with "can't start new thread".
- The optimiser backends default to one run (``n_runs=1``), matching
  ``OptimisationOptions``, instead of the machine's CPU count. Candidate
  polishing also works where the CPU count is unknown.

Organic Rankine cycle targeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New ``problem.target.carnot_orc`` (under development, not yet in the
  tutorials): parallel ORC units on the process surplus below the pinch, with
  each unit's power a second-law fraction (``ORC_ETA_II_CARNOT``) of its
  Carnot power. The search minimises the total annual cost change: ORC
  capital annualised with the capital recovery factor, less the power valued
  at ``COSTING_ORC_ELE_PRICE``, plus the change in cooling at
  ``COSTING_ORC_COOLING_PRICE``. Hot utility is unchanged and cold utility
  falls by the heat the ORC takes. Results are ``DirectOrcTarget`` objects
  (``TargetType.DORC``).
- ORC capital is ``F_inst * C_eq * (W_net / 1 MW)^n`` per unit, defaulting to
  about $3,000/kW installed at 1 MW (2025 USD) with n = 0.75, set from the US
  EPA (2021), Lemmens (2016) and Tartiere and Astolfi (2017) data; treat it
  as +/-30 %. The ORC has its own ``ORC_*`` and ``COSTING_ORC_*`` settings and
  shares nothing with heat pump targeting.
- New ``problem.target.organic_rankine_cycle`` (also under development): each
  unit is a subcritical CoolProp Rankine cycle with pump, preheating
  evaporator, optional superheat (``ORC_MAX_SUPERHEAT``), turbine
  (``ORC_ETA_TURBINE``), optional recuperator (``ORC_RECUPERATOR_ENABLED``)
  and condenser. Each fluid in ``ORC_FLUIDS`` (R1233zd(E), R1234ze(E),
  isopentane, n-pentane, toluene) is searched in turn from the Carnot design,
  evaporating at least 5 K below its critical temperature; the cheapest is
  kept. Sloped evaporator profiles are fitted under the GCC exactly.
- ORC results (net power, heat in, condenser duty, thermal efficiency,
  capital, power value, cooling and total annual cost change, fluid and
  per-unit design) appear in ``problem.results`` and the workbook. Across
  periods, thermal efficiency is total power over total heat and capital is
  the peak.
- ORC targets carry two graphs, plotted with
  ``problem.plot.grand_composite_curve_with_orc`` (the GCC before and after
  the ORC evaporators) and ``problem.plot.net_load_profiles_with_orc`` (the
  process net loads with the evaporators as a cold utility profile).
- Simulated ORC cycles keep one CoolProp state per fluid and thread, so
  ``all_periods.organic_rankine_cycle(workers=...)`` no longer lets parallel
  period solves overwrite each other's fluid state mid-calculation.

Excel and CSV inputs
~~~~~~~~~~~~~~~~~~~~

- Workbooks and CSV directories load again. The reader still added ``loc``
  and ``index`` stream columns, which the stream input contract no longer
  accepts, so every tabular source failed validation.
- Stream and utility columns missing from a sheet or CSV are left out, so the
  configured defaults apply. They were filled with 0, which set a missing
  ``dt_cont`` or ``htc`` to zero.
- ``streamlit_app.py`` runs ``problem.target.all_heat_integration()`` before
  opening the dashboard; it called the target accessor as a function.

Heat exchanger network synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Exchanger areas use the true log-mean temperature difference whenever both
  ends are positive, including an end at its minimum approach. OpenHENS fell
  back to one end there, which misstated area (for ends of 10 and 50 K, 2.5
  times too high) and with it TAC and ranking. Results now differ from
  OpenHENS for matches at the pinch.
- Each recovery match is checked against its own required approach (the sum of
  its streams' contributions), not the global dTmin.
- HEN synthesis costs are annual, as in area targeting and HPR targeting:
  utility cost is price ($/MWh) times duty over ``COSTING_ANNUAL_OP_TIME``, and
  ``COSTING_HX_*`` exchanger capital is annualised with the capital recovery
  factor. Previously prices were used as $/kW/y and capital as $/y, so default
  HEN TACs and the networks chosen change. The OpenHENS benchmark fixtures,
  which quote $/kW/y and $/y, set 1000 h/y, a zero discount rate and a
  one-year life, which reproduces their published costs.
- Multi-period networks export each stream's supply and target temperatures
  and heat load per period, and verification checks every stream's heat
  balance in every period. Multi-period networks saved without that data skip
  the stream heat-balance check, as before.
- An infeasible task outcome is marked failed and kept for diagnostics instead
  of aborting the synthesis; a utility-only network no longer stops TDM/EVM
  task building.
- Branched network evolution keeps at most ``HENS_EVM_BEAM_WIDTH`` (default 4)
  branches per depth, never solves a topology twice, and gives the same network
  in parallel and serial runs.
- The pinch design method splits each stream at the pinch plus (hot) or minus
  (cold) its own temperature contribution, not dTmin / 2 for every stream.
  With mixed ``dt_cont`` values the two sides previously moved heat across
  the pinch and broke the larger approaches.
- A problem with no possible heat recovery returns its utility-only network
  instead of failing with a division by zero in stage reduction.
- Solved networks are checked before they count as a success: every match
  against its own required approach, every exchanger's duty balance and every
  stream's heat balance to target. Duty optimisation also checks each
  exchanger's terminal approaches. A network that fails is reported as a failed
  task (``verification failed: ...``) instead of a success; solver tolerances
  are unchanged.
- A failed post-optimisation alpha solve is no longer cached: it is retried,
  and evolution ranks matches with a neutral alpha meanwhile.
- GEKKO run directories are removed once each model's results are read. A
  single default design run used to leave 25 or more behind in the temporary
  directory.

Options and exports
~~~~~~~~~~~~~~~~~~~

- Python 3.12 and 3.13 are supported again (``requires-python >=3.12``, was
  ``>=3.14.2``). The only incompatibility was Python 3.14's unparenthesised
  ``except A, B:`` syntax, now written ``except (A, B):``. CI runs the unit
  tests on 3.12 and 3.13 as well as the development version (3.14.2).

- Runtime dependencies declare tested floors: ``numpy>=2.3.2``,
  ``pint>=0.24.4``, ``pandas>=2.3.3``, ``pydantic>=2.12``, ``scipy>=1.16.1``,
  and ``CoolProp`` gains a ``<9`` ceiling. These are the oldest releases with
  Python 3.14 wheels that pass the test suite; pandas 2.3 and 3.0 are both
  supported. A new CI lane runs the unit tests at exactly these floors.

- Passing a load-time default (``THERMAL_DT_CONT``, ``THERMAL_DT_PHASE_CHANGE``,
  ``THERMAL_HTC``, ``COSTING_UTILITY_PRICE``, ``INPUT_UNIT_*`` or
  ``PROBLEM_*``) in a single ``target.*`` or ``design.*`` call's ``options``
  now raises a ``ValueError``. These values are applied when the problem is
  loaded, so the per-call value was silently ignored for the streams. Use
  ``update_options(...)`` or a workspace scenario.
- A failed Excel export to an explicit ``.xlsx`` path leaves any existing file
  in place. The workbook is written to a temporary sibling and moved into
  place only once complete; previously a failure deleted the existing file.
- ``PROBLEM_PERIOD_WEIGHTS`` (and per-value ``weights``) need either one
  weight, which weights every period equally, or one weight per period. A
  partial list now raises a ``ValueError`` naming the periods without a weight.
  Missing weights used to be filled with 1, so hour-based weights such as
  ``[6000, 2000]`` for three periods left the third with almost no weight, and
  a single weight such as ``[6000]`` became ``[6000, 1, 1]``. This replaces
  the earlier policy of padding omitted trailing weights with ``1.0``.
- ``openpinch notebook -o notebooks/`` (and ``copy_notebook(name,
  "notebooks/")``) treat a destination ending in a path separator as a
  directory and create it. A directory that did not exist yet became a file
  named ``notebooks``.
- The multiperiod weighted-average row no longer reports a pinch temperature.
  A weighted mean of the period pinches is not a temperature any period
  pinches at; each period row still has its own. Ratio rows (degree of
  integration, ETE, COP, ORC efficiency) convert every period to the first
  period's units before summing, and warn when a period has a ratio but no
  basis and is left out of the ratio of totals.
- A workbook Options sheet with only one column raises a clear
  ``ValueError`` instead of an ``IndexError``; a blank Options sheet is
  ignored.

Analysis corrections
~~~~~~~~~~~~~~~~~~~~

- Multi-start optimisation runs serially when the objective cannot be pickled
  (a lambda or closure) instead of crashing, and an error raised by the
  objective surfaces at once instead of triggering a serial rerun of every
  task. Polishing tolerates only numerical failures (``ValueError``,
  ``ArithmeticError``, ``LinAlgError``); other errors propagate.
- Utility placement's ``evaluation_limit`` is a global budget shared across
  runs and worker processes, and ``PlacementTermination.evaluations`` counts
  every worker's evaluations. 20 % of the limit is kept back for polishing the
  best candidates; the global-search runs split the rest (each run's
  ``maxfun`` is its share), and polishing also gets whatever the runs leave
  unspent. ``status`` is ``"evaluation_limit_reached"`` only when the whole
  limit has been spent. Each worker previously had the full limit, and only
  the parent's evaluations were counted.

- The cross-flow (both streams unmixed) effectiveness series in
  ``HX_Eff``, ``HX_NTU`` and ``CalcAreaUE`` included one term too few, which
  overstated effectiveness (0.556 instead of 0.476 at NTU = 1, C = 1) and
  understated area.
- Cascade Carnot heat pumps and heat engines accept a second-law efficiency of
  exactly 1. The cascade heat pump previously returned zero duty, reported as
  solved, and the heat engine was treated as plain heat recovery.
- ``compute_LMTD_from_ts`` accepts arrays (one exchanger per element); it
  raised "truth value of an array is ambiguous".
- Steam turbine work is capped at the isentropic work. An extrapolated
  correlation could report an isentropic efficiency above 1.
- The ``water.h_ps`` and ``water.s_ph`` docstrings give entropy in J/kg/K,
  the units the functions actually use; they said kJ/kg/K.
- Cascade vapour-compression cycles check that every pair of adjacent
  condenser and evaporator temperatures is in order, hottest stage first. Only
  the net change across the cascade was checked, so a stage out of order in the
  middle passed without penalty.

HPR cost model and search
~~~~~~~~~~~~~~~~~~~~~~~~~

- Vapour-compression cycles with an internal heat exchanger (IHX) compress
  from the IHX outlet state. The compressor used the IHX-outlet enthalpy with
  the evaporator-outlet entropy, which understated its work so heating COP
  rose with IHX duty and could exceed the Carnot limit. Simulated VC,
  parallel, cascade and VC-MVR results change, and the search no longer
  favours large IHX duties.
- Simulated HPR targets now carry installed capital per machine:
  ``C = F_inst * C_eq * (Q_cap / 1 MW)^0.7 * (0.7 + 0.3 * n_stages) * f_T``,
  with ``C_eq`` = $485k fitted to the IEA HPT Project 68 cost data (2025 USD),
  ``F_inst`` = 2.3 and a temperature factor above 75 °C. Parallel cycles are
  separate machines; cascades and VC+MVR systems are one.
- Electricity and heat default to $100/MWh. Ambient air is a free utility
  inside each HPR cascade; cooling water (25 °C, 5 K approach) and a default
  refrigeration utility (0.4 of the Carnot COP) cover the remaining cooling.
- Default-utility capital ($750/kW of heating, $1500/kW of refrigeration) is
  annualized into the objective. ``COSTING_HPR_CAPITAL_RECOVERY_ENABLED`` and
  ``COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED`` switch each part off.
- Breaking: ``COSTING_HPR_COMP_*``, ``COSTING_HPR_HX_DUTY_*`` and
  ``COSTING_HPR_PRICE_RATIO_COLD_TO_ELE`` are removed, and the
  compressor/heat-exchanger capital split is gone from results.
- Stage duties are independent fractions of the available duty, so the search
  has no flat regions from clipped requests. A heat pump that does not pay
  raises a typed "no beneficial heat pump" error.
- Simulated refrigeration penalises unserved selected cooling, as the Carnot
  path does.
- Parallel vapour-compression refrigerators send all their condenser heat to
  the process or ambient air, as cascades already did. Each cycle used to place
  only a 1 kW placeholder, so the rest of its heat was rejected at any
  condensing temperature for free.

HEN duty allocation on a fixed structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``problem.design.optimise_duties(structure, objective=...)`` allocates duty
  on a user-defined network (recovery matches by stage plus heaters and
  coolers, given as a mapping or a ``HeatExchangerNetwork``) with fixed
  matches and free stream splits. Objectives: ``"utility"`` (minimum approach
  per exchanger), ``"area"`` (common area across periods under hot and/or
  cold utility caps) and ``"cost"`` (total annual cost, 1 K feasibility
  approach).
- A stream can have several utility exchangers, one per utility and position,
  placed at the stream end or between stages.
- Exchangers left at zero duty are removed one at a time and the problem is
  re-solved; the result lists them.
- Segmented streams and utilities are supported: piecewise profiles, the
  minimum approach at every segment boundary inside an exchanger, and areas
  from duty-aligned slices.
- The network grid draws every utility exchanger at its own position (stream
  end or stage boundary) instead of merging a stream's utility exchangers.

Analysis reliability and extensibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Default hot and cold utilities now cover the whole demand in every zone
  when zones have different ``dt_cont`` multipliers. They were placed for the
  root zone's multiplier and before the zone multipliers were applied, so a
  zone with a larger multiplier left part of its duty unmet (for example 100
  of 300 kW). They are now placed for the largest multiplier in the tree.
- The unmet-utility warning ignores residues below a millionth of the
  problem's process duty, so cascade rounding no longer reports a few watts
  as unmet.
- New ``REPORTING_DEBUG_ENABLED`` option (default false). Repeated stream
  names within a zone are still renamed ``_1``, ``_2``, but the warning is
  reported only in debug mode.
- Made golden HPR contract fixture provenance independent of the installed
  release, preventing automatic version bumps from making fixtures stale.
  Calculated performance maps continue to record their runtime versions.
- Scoped published rows and graphs to the requested zone and traversal,
  including successive all-period batches for different sibling zones.
- Included the effective HPR simulation backend in target provenance so
  backend changes invalidate superseded ``base_target`` references.
- Detached nested HPR performance-map provenance observations to preserve
  validated maps and exports without changing the schema 1.0 JSON boundary.
- Unified serial and parallel runtime snapshots, preserving process-MVR stream
  references and compressor work. Target execution and complete period batches
  commit only after success, with prepared period state retained for enrichment.
- Fixed period-row relabeling, invocation-specific exergy settings, foreign-zone
  ownership, and selected-zone traversal. Replacing prerequisites invalidates
  dependent targets; component and input changes clear dependent caches.
- Added immutable provenance and method metadata, family-owned reporting adapters,
  and complete aggregation policies. Weighted rows exclude period-specific HPR
  simulation records. Brayton fails before state preparation.
- Observation properties now return detached snapshots or read-only mappings.
  Stale and foreign ``base_target`` references are rejected. See
  :doc:`overview/analysis-migration` for the intentional API changes and
  :doc:`developer/adding-analysis-methods` for the extension contract.

HPR and MVR robustness and tutorial reliability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Hardened CoolProp-backed HPR residual accounting when shared temperature
  grids contain duplicate levels. Equivalent detached residual rows collapse
  deterministically, while conflicting duplicates and unalignable grids fail
  explicitly instead of corrupting HPR utility profiles.
- Added a standard-corpus end-to-end benchmark with 54 distinct assignments:
  18 vapour-compression heat-pump, 18 vapour-compression refrigeration, and 18
  optimized VC+MVR cases. Bounded typed failures remain atomic, and declared
  sentinel cases must show convergence toward the selected finite objective.
- Rebuilt tutorials 08 through 11 with explicit topology and search limits,
  compact plain evidence, direct required CoolProp solves, distinct optional
  TESPy and Brayton statuses, and separately attributable execution tests.
- Notebook 10 now demonstrates five separate shared-design optimizations over
  ``turndown``, ``base``, and ``peak`` using fresh problems, followed by one
  more tightly bounded optional advanced cascade. It no longer presents
  independent ``target.all_periods`` replay as shared installed-design
  optimization.
- Expanded the fundamentals, workflow guide, public HPR reference, capability
  matrix, support boundary, notebook catalog, and generated coverage metadata
  to document budgets, diagnostics, direct process MVR, optimized VC+MVR, and
  the two multiperiod modes.

Target-owned HPR performance maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added ``simulation_backend="coolprop"`` to the existing
  vapour-compression heat-pump and refrigeration target methods. Omission keeps
  the unchanged CoolProp path; selecting ``"tespy"`` uses optional TESPy design
  solves to rank supported scalar single-stage target candidates, with no
  CoolProp fallback.
- Successful compatible targets retain a frozen, versioned-data-friendly
  ``HprTargetSimulationRecord``. The explicit
  ``problem.target.hpr_performance_map(target=..., request=...)`` operation uses
  that target-owned backend and nominal basis to generate one atomic physical
  map. TESPy map points use a winning-design/offdesign lifecycle.
- Pure fluids, registered blends, and explicit N-component molar mixtures are
  capability-checked at dew-point evaporation and bubble-point condensation
  states. ``REFPROP`` and unsupported topology or state combinations fail
  before optimization.
- The map remains a plain schema ``1.0`` boundary for OpenUtility and other
  MILP consumers. OpenPinch imports neither OpenUtility, Pyomo, nor HiGHS;
  downstream consumers own capacity scaling, adjacent-segment interpolation,
  commitment, electricity coupling, and multiperiod dispatch.
- Selected scalar periods and sequential independent all-period replay are
  supported. Shared-vector multiperiod TESPy targeting, multi-port flattening,
  automatic grids, partial maps, and a thread-safety promise are not included.

Heat-recovery ``dt_min`` service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added non-mutating selected-period and all-period inverse pinch targeting
  through ``target.heat_recovery_dt_min``. The service returns
  the global ``dt_min`` corresponding to requested process heat recovery,
  validates the zero-``dt_min`` thermodynamic limit, supports explicit units,
  and mirrors through active workspaces and isolated ordered case batches.
- Added deterministic plateau and zero-recovery boundary handling, analytical,
  packaged-regression, immutability, and seeded property-based coverage.
- Hardened the service after a full edge-case audit: scalar inputs and
  low-level numerical arguments now reject implicit coercions; inverse cascade
  boundaries preserve exact shifted temperatures to ``1e-6 delta_degC``;
  foreign ``Zone`` selectors resolve locally by address; positive micro-duty
  requests are distinct from exact zero; result contracts validate units and
  cross-field relationships; and exact all-period IDs take precedence over the
  ``value``/``unit`` scalar-mapping shape.
- Corrected thermodynamic-limit inversion for threshold problems: maximum
  recovery now returns the greatest positive global ``dt_min`` that retains the
  limit, rather than forcing zero ``dt_min``.
- Documented the distinction between process composite-curve global
  ``dt_min`` and exchanger-level EMAT, and extended tutorials 02 and 06 with
  selected-period
  and scalar/mapped all-period workflows.
- Added a dedicated inverse-``dt_min`` task guide and integrated the workflow
  through RTD navigation, overview, fundamentals, public problem/workspace API,
  contributor service reference, and notebook-series pages. Tutorial 02 now
  demonstrates explicit recovery units, every result field, and preservation
  of the ordinary cached target.

0.5.0 architecture clean break
------------------------------

- ``PinchProblem`` remains the strict
  mapping-in/result-out contract. Its signature, validation order, return
  structure, serialization, ordering, exceptions, and numerical behaviour are
  unchanged.
- The package root exports exactly ``PinchProblem`` and ``PinchWorkspace`` as
  the high-level workflow entry points. Schemas, enums, resource helpers, and
  the main service remain with their concrete owners.
- Business state, wire contracts, reusable optimisation, orchestration,
  engineering analysis, infrastructure adapters, and presentation now have
  explicit ``domain``, ``contracts``, ``optimisation``, ``application``,
  ``analysis``, ``adapters``, and ``presentation`` owners.
- ``OpenPinch.classes``, ``OpenPinch.lib``, ``OpenPinch.services``,
  ``OpenPinch.utils``, and ``OpenPinch.streamlit_webviewer`` are removed.
  There are no aliases, forwarding facades, dynamic export barrels, pickle
  shims, or migration package. Old deep imports and Python pickles are not
  migrated.
- Runtime stream segments, exchanger period states, exchanger area slices,
  process-MVR records, multiperiod HPR cases, dashboard state, graph build
  records, and HEN solver state remain private to their parent or service
  owner.
- Optimisation is a reusable package-level capability. Heat-pump targeting
  crosses one explicit adapter; new services can reuse optimisation without
  importing heat-pump code.
- HPR internal records are attribute-only, and HPR optimiser identifiers must
  exactly match ``dual_annealing``, ``cmaes``, ``bo``, or ``rbf_surrogate``.
  Obsolete helper-signature retries and historical optimiser spellings are not
  accepted.
- ``StreamCollection`` supports current-version pickle round trips, including
  deterministic fallback for an unpicklable callable sort key. It does not
  repair state dictionaries produced by older package versions.
- Internal type-only imports now resolve to the concrete configuration owner,
  total-site utility profiles forward the canonical period keyword, and invalid
  crossflow row counts raise an explicit validation error.
- Tests now mirror observable owner layers. The external main contract is
  authoritative, architecture directions are AST-enforced, and CI measures
  seeded statement and branch coverage with Hypothesis seed ``20260715``.

This is an intentional pre-1.0 clean break. No data migration, import migration,
pickle migration, or compatibility layer is provided.

Domain and input contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~

This pre-release changes the following contracts without compatibility shims:

- Values owned by a ``Stream`` and its segment records are read-only. Mutations use
  explicit stream assignment, indexed-value, or segment-update APIs.
- Period weights use one validation policy: omitted trailing weights become
  ``1.0``; excess, non-finite, negative, and all-zero vectors are rejected.
- Structured process-stream and nested thermal-profile inputs reject unknown
  fields. Process streams accept the canonical ``name`` and
  ``heat_capacity_flowrate`` spellings only.
- Workspace bundles use schema version ``2`` and ``case_input``. Version ``1``,
  unknown versions, and the retired ``payload`` field are rejected.
- Segmented process streams and utilities share the same semantic validation in
  reports and preparation, including parent aggregate consistency.
- ``TargetInput.network`` accepts the exact mapping produced by
  ``HeatExchangerNetwork.model_dump(mode="json")`` and preserves it through a
  complete input JSON round trip. The transport contract rejects private
  solver and source metadata and does not automatically seed HEN synthesis.
- HEN endpoint classifications now use ``StreamID`` values ``Process`` and
  ``Utility``. The retired lowercase endpoint-role enum and ``Unassigned``
  endpoint values are not accepted.
- ``StreamID`` is string-backed, keeping Python-mode HEN and canonical input
  mappings safe for direct JSON and workspace-bundle persistence.

Private helper ownership
~~~~~~~~~~~~~~~~~~~~~~~~

- Private class helpers are grouped under owner-oriented packages for streams,
  values, collections, problem tables, problem orchestration, workspaces, and
  heat exchangers.
- Runtime stream-segment, exchanger-period, and exchanger-area-slice record
  classes are parent-owned implementation details. Construct them through
  ``Stream`` and ``HeatExchanger`` mappings. The equivalent segment wire shape
  remains accepted through ``PinchProblem(...)``; direct schema
  imports are not compatibility-protected.
- The former public record imports and their Python pickle paths are removed
  without aliases or compatibility shims.
- Synthesis schemas now have concrete common, topology, method, task, and result
  owners. The compatibility-only ``methods``, ``tasks``, and ``results`` modules
  and the synthesis package re-export barrel are removed. Import concrete owner
  modules instead; old barrel-qualified pickle paths are unsupported.
- Owner package ``__init__`` files remain import-free markers. The package root
  is the deliberate two-class workflow surface. The retired ``classes``,
  ``lib``, ``services``, ``utils``, and
  ``streamlit_webviewer`` package paths have no compatibility facades; advanced
  callers import concrete owner modules.
- Process-MVR records, multiperiod HPR period cases, dashboard graph state,
  graph specifications/metadata, and HEN solver runtime records are private to
  their owning services. Concrete parent components, schemas, direct-MVR
  models, and HEN equation-model classes remain available as unsupported
  internals.

Period-native PDM and utility constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- PDM decompositions expose ordered ``period_targets`` with explicit period
  identities and indices. Clipped temperatures and active-stream flags are
  period-indexed; the retired singular target and clipped-temperature fields do
  not exist.
- Each operating period uses its own utility targets and pinch temperature.
  Shared topology is the union of streams and matches active in any period, and
  a pinch side is solved when any period requires it.
- Above- and below-pinch amalgamation retains every period's duties,
  temperatures, approach variables, split fractions, and explicit
  non-isothermal branch outlet temperatures.
- Non-isothermal warm starts normalize hot split fractions across cold matches
  and cold split fractions across hot matches independently in every period.
- Segmented utilities use local per-segment ``dt_cont`` values. The inlet uses
  the first segment contribution, the solved match outlet uses the traversed
  segment contribution, and an exact boundary uses the larger adjacent value.

Period-native HEN results
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``HeatExchanger`` retains shared topology, design area, and capital fields.
  Operational duty, activity, approaches, split fractions, and source/sink
  temperatures are stored in non-empty ordered ``period_states`` containing
  parent-owned period-state records.
- The retired exchanger-level operating scalar fields do not exist. Use
  ``exchanger.state(period_id)``; omission is accepted only for an exchanger
  with exactly one period state.
- Multiperiod duty, temperature, diagram, export, and controllability queries
  require ``period_id``. No implicit period-zero selection is provided.
- Extraction walks every period array, retains matches active only outside the
  first period, and prefers explicit non-isothermal branch outlet temperatures.
  Solved branch split fractions keep downstream duty checks physically valid.

Isolated summaries and HPR economics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Multiperiod summary replay captures one baseline zone and solves every period
  against a fresh deep copy. The original zone object, cached results, and
  recorded targeting specification are restored after success or failure.
- Shared simulated-HPR candidates are ranked by weighted operating cost plus
  weighted feasibility penalty plus maximum annualized capital cost. Weighted
  backend ``obj`` is used only for backends that provide no cost breakdown.
- Weighted internal HPR summaries average operating fields, use the maximum
  capital, annualized-capital, compressor-capital, and heat-exchanger-capital
  fields, and recompute total annualized cost as weighted operating cost plus
  maximum annualized capital. Non-HPR fields retain weighted averaging.
