Pre-Release Notes
=================

Unreleased
----------

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
