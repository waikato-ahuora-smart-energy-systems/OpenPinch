:orphan:

October 2026 Audit Fix Plan
===========================

This plan schedules the fixes for the second audit of OpenPinch ``develop``
(``0dd0caba``, version 0.7.2 dev, 2026-10-19), and for the first audit's items
that are still open. Both audit reports are kept outside the repository. The
plan groups the findings into work packages, each one pull request into
``develop``, and fixes their order and the conventions the fixes follow.

The finding ids (C1, H1, M1 and so on) are the second report's. First-audit
ids are prefixed ``A1-``.

Security findings are described here only by what changes. Their reproducers
stay in the report until the fix is released.

Scope and Baseline
------------------

What the audit found correct, and what this plan must not disturb:

- Hot and cold utility targets and pinch temperatures agree with an
  independent Problem Table on 750 random problems, every sample case and
  every period of the multiperiod cases.
- Composite, shifted, Grand Composite and Total Site profiles agree with
  independent constructions.
- Exchanger duties, LMTD and area in synthesised networks agree with an exact
  recomputation to within 0.02 percent.
- Vapour-compression cycles agree with a CoolProp recomputation to 1e-6.

The defects sit in quantities built on those targets: area, unit count,
utility cost, Heat Pump placement, HEN synthesis, multiperiod summaries, and
input and export handling.

Decisions
---------

These settle the questions of intent the report left open.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Finding
     - Decision
   * - C2, M3
     - The Heat Engine and recovery branch of the Carnot cascade only moves
       heat downhill, from an evaporator to a condenser at or below its
       temperature, pair by pair. With ``HPR_INTEGRATED_EXPANDER_ENABLED`` off
       the pair is plain heat recovery with no work credit.
   * - C4
     - Area targeting uses the left-limit driving force at every slice end.
       The discontinuity rule is removed from the area and utility placement
       paths. It stays only in the Temperature Driving Force plot, if that
       plot needs it.
   * - M1
     - ``hpr_cop`` follows the duty: Heat Pumps report the heating COP
       (Q_cond/W), refrigeration reports the cooling COP (Q_evap/W). The
       field help says which.
   * - M2
     - The Carnot heating COP stays 1 + eta x T_l/(T_h - T_l): the
       second-law efficiency applies to the cooling Carnot COP, and the
       heating COP follows from the energy balance. ``HPR_ETA_II_CARNOT``
       help and the Heat Pump guide are corrected to say so.
   * - M9
     - ETE becomes the exergy transfer effectiveness of the Balanced
       Composite Curves: the exergy gained by the cold balanced composite
       over the exergy released by the hot balanced composite, both relative
       to ``ENV_TEMPERATURE`` and with utilities included. It is bounded by
       100 percent, and the shortfall is the exergy destroyed in heat
       transfer.
   * - H5
     - Per-period values without ``PROBLEM_PERIOD_IDS`` are a validation
       error.
   * - H6
     - HEN synthesis models every hot and cold utility: the superstructure
       gets a utility index. This is the largest package in the plan.
   * - H8
     - With no method given, synthesis keeps quality tier 0 and runs one set
       of tasks for every approach temperature. The manifest reports what
       ran.
   * - M7
     - The accepted network is the feasible network with the lowest total
       annual cost across all method stages, and the result records the
       stage it came from.
   * - M12
     - When the main module has no ``if __name__ == "__main__"`` guard and
       workers would be spawned, utility placement runs serially and warns
       how to enable workers.
   * - M13
     - A negative ``dt_cont`` stays a warning. The warning says the reported
       heat recovery can then exceed the thermodynamic limit.
   * - M14
     - ``heat_capacity_flowrate`` is supported: without ``heat_flow``, the
       heat flow is CP x abs(T_supply - T_target), non-negative for hot and
       cold streams alike. When both are given and disagree, validation
       reports an issue.
   * - M15
     - With assisted heat transfer, unmet pocket demand below the pinch is
       met by hot utility and reported, so the energy balance always closes.
   * - M21
     - Annual utility cost uses ``COSTING_ANNUAL_OP_TIME``, not a calendar
       year.

Conventions for Every Package
-----------------------------

- Strict TDD as in ``AGENTS.md``: the failing test first, the test and its
  fix in one commit, and a release note under "Unreleased".
- Tolerances are relative to the problem's duty scale (as ``pinch_idx``
  already does), not absolute kW. A test that pins a tolerance records why.
- Independent oracles become regression tests, not one-off scripts:

  - a plain Problem Table for utility targets;
  - a fine-slice Bath integration for area (C4);
  - entropy generation, sum(Q/T) out minus in, for Heat Pump and Heat Engine
    targets (C2);
  - N = S - 1 per pinch region for the unit count (H1);
  - per-period single-period problems for multiperiod rows.

- A change to published numbers names the old and new values for the
  tutorials in its release note, and re-records the OpenPinch-WASM fixtures
  for the affected tutorials.
- Each package runs the unit lane and, where it touches synthesis, the solver
  lane before its pull request.

Release Sequence
----------------

1. **0.7.2, security.** Work package 1 alone, released as soon as it merges.
2. **Then one patch release per work package**, in the order below, as each
   merges. Packages 2, 3, 4, 8 and 11 change published numbers; each
   re-baselines once, in its own release.
3. Each release pull request moves the "Unreleased" release notes under the
   version heading with its date and starts a new "Unreleased" section
   (A1-M7). The first such pull request also files the 0.7.0 and 0.7.1
   notes under their versions.

Work Packages
-------------

1. Input and Engine Hardening (0.7.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: C1, M25, M26, H10.

- Solver options (C1): ``HENS_SOLVER_OPTIONS_*`` accept only an allow-list
  of option names (tolerances, iteration limits and numeric settings) with
  numeric or simple-token values. Anything else is a validation issue, and
  option files are written only from validated values.
- Unit strings (M25): unit text is parsed by a bounded grammar before it
  reaches pint, with a length limit and no exponent chains.
- Batches (M26): ``workspace.batch`` cannot nest, or nests to a fixed depth,
  with a cap on the commands one request may run.
- Exports (H10): text cells that start with ``=``, ``+``, ``-``, ``@``, a
  tab or a carriage return are written as plain strings in every Excel
  export (summary, problem tables, dashboard exports).

App follow-up (OpenPinch-WASM, needs its own change): the backend binds to
localhost only, and its job API gets request size and time limits.

2. Heat Pump Second Law
~~~~~~~~~~~~~~~~~~~~~~~

Findings: C2, M3, and the low item that Carnot targeting never checks
``cycle.solved``.

- Allocate Heat Engine and recovery duty over the ``is_on``
  condenser-evaporator pairs, not over the unions of their indices.
- A pair whose pooled temperatures give T_h <= T_l gets no duty.
- ``solved`` requires entropy generation >= -tol, and targeting checks it.
- ``HPR_INTEGRATED_EXPANDER_ENABLED`` decides whether a pair earns work, as
  in the decision above.
- Re-check cascade vapour-compression targeting, which seeds from this
  optimiser.
- Tests: the entropy-generation oracle on ``basic_pinch`` and ``pulp_mill``
  (both currently negative), and 1x1 and parallel cases unchanged.

Published numbers: default cascade Carnot targets on the sample cases change.

3. Area and Tolerance Correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: C4, H1, H3, M24, A1-M13, and the low curve items (the always-False
guard in ``intervals.py``, ``np.isclose`` trimming real curve ends, empty
series, both pinch temperatures None on a balanced problem).

- Remove the discontinuity rule from area targeting and from utility
  placement's use of the same driving-force function (C4).
- Flatten the inserted pocket-exit row in
  ``_remove_pockets_on_one_side_of_the_pinch``, or insert the unrounded
  temperature (H3).
- Make the ``area_cost`` balance check and the ``num_units`` boundary test
  duty-relative, and always count the first and last rows as boundaries (H1).
- Segment slicing tolerates solver overshoot relative to the duty (A1-M13).
- Tests: the Bath oracle on Yee and Grossmann (337.20 m2), ``pulp_mill`` and
  random cases; N = S - 1 on every sample case; the scaling sweep from x1e-6
  to x1e3 and a +100 K shift (M24).

Published numbers: area and capital cost on cases whose composite curves
have a temperature gap (Yee and Grossmann 341.08 to 337.20 m2), unit counts
on ``basic_pinch`` (0 to 7), ``crude_preheat_train`` and
``chocolate_factory``. This is the second area change in two releases, so
the release note explains both.

4. Utility Cost and Totals
~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: H2, M22, M21.

- Direct targets compute ``utility_cost`` with ``calc_utility_cost``.
- Total Site utility lists and their cost come from the cascaded site
  result, so they agree with the Total Site targets.
- The area target's ``total_cost`` is annualised capital plus utility cost.
- Annual cost uses ``COSTING_ANNUAL_OP_TIME``.
- Tests: ``basic_pinch`` utility cost 17.5 $/h; a Total Site case whose
  listed utilities sum to its targets.

Published numbers: utility cost (from 0), total cost and annual figures.

5. HEN Utility Approach Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: C3, and the low item that ``verify()`` compares the utility cost
with itself.

- Terminal approach constraints on every heater and cooler, segmented or
  not, with the same big-M pattern as the segmented branch.
- Verification checks utility approaches and compares utility cost with an
  independent recomputation.
- A crossed exchanger gets no fallback LMTD area; it is a verification
  failure.
- Tests: the audit's cross case (cooling water at 320 to 325 K) is rejected
  or solved without a cross; solver-lane runs of the tutorial networks are
  unchanged.

6. HEN Utilities and Multiperiod Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: H6, H7.

- Index the utility axis through the stagewise, PDM and TDM models: setup,
  objectives, post-processing and area (H6).
- Gate PDM approach constraints, theta and area on the period's activity,
  and keep inactive streams out of post-processing (H7).
- Document multi-utility synthesis in the HEN guide.
- Tests: ``dairy_chilling_multiperiod`` synthesises with its chilling
  coolers on the chilled utility; a two-utility case picks the cheaper
  feasible utility; ``crude_preheat_train_multiperiod`` solves at dTmin 5,
  10 and 20.

Size: the largest package. If it runs long, H7 can merge first on its own.

7. HEN Orchestration
~~~~~~~~~~~~~~~~~~~~

Findings: H8, M5, M6, M7, A1-H8, A1-H10, and the low item that
``heat_exchanger_network(initial_networks=...)`` always raises.

- Tier 0 runs every approach temperature; the manifest reports what ran (H8).
- Split branches report their own outlet temperatures and area (M5).
- Multiperiod stage reduction reads every period (M6).
- The accepted network is the lowest-TAC feasible one across stages (M7).
- Piecewise linearisation sorts by x before refining (A1-H8).
- The single-stage stagewise gamma_c is 0 (A1-H10).
- ``initial_networks`` either works or leaves the signature.

8. Multiperiod Correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: H4, H5, M17, M18, and the low items on
``turbine_efficiency_target`` averaging and ignored ``period_ids`` keys.

- Select placement outputs and weights by period id, and let a period subset
  summarise, report and export (H4).
- Per-period values without ``PROBLEM_PERIOD_IDS`` are a validation error,
  and records of different lengths get a clear message (H5).
- ``StreamSegment.set_attr_for_period`` broadcasts a scalar only to the named
  period, as the stream fix did in 0.7.1 (M17).
- A single-period call finds that period's retained state after an
  all-periods run (M18).
- ``turbine_efficiency_target`` is total work over total heat, not a mean of
  ratios; domain objects reject a ``period_ids`` key.

9. Input, Contracts and Exports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: H11, M14, M19, M20, M23, A1-M10, and the IO low items.

- Blank and whitespace rows are dropped before cells are wrapped (H11).
- CSV and Excel loaders read the header row and map columns by name (M23).
- ``heat_capacity_flowrate`` is supported as decided above (M14).
- The Excel input template's option names load, or the template is updated
  to current names (M19).
- ``compare_to`` works across different project names (M20).
- Configuration metadata reports numeric lists as numeric and the minimum of
  ``positive=True`` fields as exclusive (A1-M10).
- Low items: warn on unnamed utility rows, reject names that normalise to a
  duplicate, read TRUE cells as booleans, and allow "/" in project names.

10. Optimisers and Utility Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: H9, H12, M8, M10, M11, M12, A1-M14, and the low placement and
pool items.

- Clamp the sensible-level span bound so the lowest reachable temperature
  stays positive (H9). The Yee and Grossmann case places with sensible
  levels.
- Polished candidates are compared on feasibility first, then objective
  (H12).
- Every local method polishes within the bounds, and ``local_method`` is a
  closed choice (M8).
- Placement uses ``ENV_TEMPERATURE`` (M10).
- The entropy breakdown uses one basis throughout (M11).
- An unguarded main module runs serially with a warning (M12).
- Optimiser options are validated before the solve (A1-M14).
- The worker pool has an Emscripten guard.
- The optimised placement case keeps the original utilities' price, htc and
  ``dt_cont``. "Min Exergy Destruction" is renamed to what it measures.

11. Heat Pump Conventions, Exergy and Targeting Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Findings: M1, M2, M4, M9, M13, M15, M16, and the vapour-compression
docstring item.

- ``hpr_cop`` by duty (M1), and the Carnot COP definition documented (M2).
- Zero-lift vapour-compression stages keep their full duty (M4).
- ETE on the Balanced Composite Curves (M9).
- The negative ``dt_cont`` warning states its consequence (M13).
- Assisted heat transfer meets pocket demand below the pinch with hot
  utility (M15).
- Total Site stops rounding its Problem Table in place, and the energy
  transfer diagram is built from unrounded data (M16).
- Docstrings: dew and bubble the right way round for T_evap and T_cond.

Published numbers: refrigeration COP, ETE, and ``zonal_site`` utilities.

12. Command Layer
~~~~~~~~~~~~~~~~~

Findings: the command-layer low items.

- Warnings are collected per call, so concurrent ``run_command`` calls keep
  their own warnings and leave the global filters unchanged.
- ``OrcTargetingError`` returns an ``error`` envelope.

Not Covered by Either Audit
---------------------------

A later pass should cover:

- MVR, Brayton and TESPy;
- multiperiod HEN synthesis beyond PDM;
- cross-period leakage for Heat Pumps, energy transfer and cogeneration;
- the Medina-Flores turbine coefficients;
- ``np.interp`` monotonicity at the air-source Heat Pump sites.

Tracking
--------

.. list-table::
   :header-rows: 1
   :widths: 6 38 30 14 12

   * - WP
     - Package
     - Findings
     - Numbers change
     - Status
   * - 1
     - Input and Engine Hardening (0.7.2)
     - C1, M25, M26, H10
     - no
     - planned
   * - 2
     - Heat Pump Second Law
     - C2, M3
     - yes
     - planned
   * - 3
     - Area and Tolerance Correctness
     - C4, H1, H3, M24, A1-M13
     - yes
     - planned
   * - 4
     - Utility Cost and Totals
     - H2, M21, M22
     - yes
     - planned
   * - 5
     - HEN Utility Approach Constraints
     - C3
     - possibly
     - planned
   * - 6
     - HEN Utilities and Multiperiod Synthesis
     - H6, H7
     - possibly
     - planned
   * - 7
     - HEN Orchestration
     - H8, M5, M6, M7, A1-H8, A1-H10
     - possibly
     - planned
   * - 8
     - Multiperiod Correctness
     - H4, H5, M17, M18
     - yes
     - planned
   * - 9
     - Input, Contracts and Exports
     - H11, M14, M19, M20, M23, A1-M10
     - no
     - planned
   * - 10
     - Optimisers and Utility Placement
     - H9, H12, M8, M10, M11, M12, A1-M14
     - possibly
     - planned
   * - 11
     - Heat Pump Conventions, Exergy and Targeting Details
     - M1, M2, M4, M9, M13, M15, M16
     - yes
     - planned
   * - 12
     - Command Layer
     - low items
     - no
     - planned
