Heat Pump and Refrigeration Methods
===================================

OpenPinch separates engineering model choice into descriptive methods:

``carnot_heat_pump`` and ``carnot_refrigeration``
   Fast thermodynamic screening. Use utility-placement and cascade-topology
   booleans to express the physical arrangement.

``vapour_compression_heat_pump`` and ``vapour_compression_refrigeration``
   Refrigerant-specific simulated cycles with explicit refrigerant candidates.

``brayton_heat_pump`` and ``brayton_refrigeration``
   Brayton-cycle studies. These raise ``NotImplementedError`` until the
   Brayton solver is complete.

``mvr_heat_pump``
   Vapour-compression/MVR cascade targeting.

The method name selects the model; configuration does not. Named
``load_fraction``, ``load_duty``, or ``period_loads`` values avoid a separate
load-mode string. Only one load form may be supplied per call.

Placement and Topology
----------------------

Direct placement couples the HPR model to a process heat-integration target.
Utility placement couples it to the site utility system.
``is_utility_heat_pump`` (heat pump methods, including ``mvr_heat_pump``) or
``is_utility_refrigeration`` (refrigeration methods) selects utility
placement; the default ``False`` is direct placement.

Cascade topology, the default (``is_cascade_cycle=True``), connects successive
temperature lifts. Pass ``is_cascade_cycle=False`` for parallel, independent
lifts. The flag applies to the Carnot and vapour-compression methods. MVR is a separate named workflow rather than a hidden
cycle-string combination.

In a cascade each lower stage condenses into the evaporator of the stage above
it. With one condenser and several evaporators, only the top stage delivers
process heat; the lower stages lift heat to the stage above. In multi-stage
refrigeration each stage's condenser passes heat up to the evaporator of the
warmer stage, and the top stage rejects the remainder. A stage's subcooling heat
must fit in its own process duty, so a stage that only passes heat up does not
subcool.

A CoolProp stage may condense above its fluid's critical temperature. It then
runs transcritical: its high pressure is the critical isochore's pressure at
the condensing temperature, and its gas cooler glides down to that temperature
less the subcooling. Its pressure is still held to the discharge pressure
limit below. The warm start is always subcritical. TESPy targets stay
subcritical.

Each vapour-compression stage is held to practical limits. Condensate is
injected into the compressor discharge until the gas is at most
``HPR_MAX_DISCHARGE_SUPERHEAT`` (default 50 K) above its dew point; the
condenser rejects the same heat and the compressor does the same work. A
stage is penalised, by its condenser duty times the relative violation, when
its pressure ratio is above ``HPR_MAX_PRESSURE_RATIO`` (default 15), its
discharge pressure above ``HPR_MAX_DISCHARGE_PRESSURE`` (default 60 bar;
``HPR_MAX_DISCHARGE_PRESSURE_CO2``, 140 bar, for carbon dioxide), its suction
pressure below ``HPR_SCREEN_MIN_SUCTION_PRESSURE`` (default 0.5 bar), or its
lift below ``HPR_MIN_STAGE_LIFT`` (default 5 K), and whenever the compressor
discharges two-phase. MVR stages on process vapour are not held to these
limits.

Ambient air is a free heat source for heat pumps and a free heat sink for
refrigeration unless ``HPR_AMBIENT_AIR_ENABLED`` is ``False``.

Bounded Search and Failure Evidence
-----------------------------------

HPR targeting accepts explicit ``maximum_restarts``, ``maximum_iterations``,
and ``maximum_evaluations`` limits. These are work bounds, not promises that an
arbitrary topology will converge. Start with a one-condenser,
one-evaporator configuration and one MVR stage where applicable, prove that the
required service succeeds, and only then increase topology complexity or search
allowances.

A successful target has ``hpr_success`` set and exposes finite accounting in
``hpr_details``. A bounded search that finds no valid candidate raises
``HPRTargetingError`` with an ``HPRFailureSummary`` containing the configured
budget, evaluated-candidate count, category counts, and at most sixteen
representative failures. ``diagnostics.model_dump(mode="json")`` is the stable
plain-data inspection boundary.

Calls required by an engineering workflow should be allowed to fail loudly.
An optional dependency or exploratory method may instead handle
``ImportError``, ``NotImplementedError``, or ``HPRTargetingError`` separately.
Do not catch a broad ``Exception`` or relabel an unavailable TESPy or Brayton
calculation as a CoolProp success.

Thermodynamic Backend Selection
-------------------------------

The current ``vapour_compression_heat_pump`` and
``vapour_compression_refrigeration`` methods are the targeting tie-in point.
CoolProp is the default. Selecting TESPy explicitly makes each unique optimizer
candidate use a fresh design solve, and the accepted TESPy duties, power, COP,
and thermal profiles determine target ranking and the returned target. There is
no CoolProp fallback, retry, or fabricated successful value after a TESPy
candidate failure.

The initial TESPy target model is scalar and single-stage: one evaporator, one
condenser, and one refrigerant loop. It excludes parallel or multi-port cycles,
integrated expanders, an internal heat exchanger, and shared-vector
multiperiod optimization. Candidate-local convergence failures allow later
candidates to run; dependency, configuration, lifecycle, and cleanup failures
abort the call. Targeting is sequential and each call owns its evaluator and
exact bounded cache.

Working Fluids and Saturation Anchors
-------------------------------------

Both backends accept installed-property pure fluids, registered blends, and
explicit N-component molar mixtures; the explicit syntax is
``HEOS::component[fraction]&component[fraction]``. Fractions are molar,
normalized without changing component order, and preserved in target and map
provenance. OpenPinch does not impose a component-count allowlist.

Capability is checked at the actual states. Evaporation pressure uses the dew
point and condensation pressure uses the bubble point, which matters for
zeotropic blends. Preflight checks one evaporating and condensing probe pair for
each stage, so a fluid that passes it can still be unsupported at other states the optimizer tries; those candidates are rejected
during optimization. The external ``REFPROP`` backend is outside this contract.

Refrigerant Screening
---------------------

``problem.target.hpr_refrigerant_screen`` checks which working fluids suit the
levels a Carnot target places before a vapour-compression study. A Carnot
target supplies its distinct levels whose ports carry duty and its topology,
parallel or cascade, and whether it is a heat pump or refrigeration; explicit
``T_evap`` and ``T_cond`` can be given instead, with ``topology`` and, for
refrigeration, ``is_refrigeration=True``. The levels are saturation
temperatures, taken hottest first.

The screen lists every layout the levels allow. Each choice of ``n``
condensing and ``m`` evaporating levels, from one of each up to all of them,
is a layout of the targeting method that would use it:

- one condensing and one evaporating level: a single-stage cycle;
- parallel topology: ``max(n, m)`` parallel machines, machine ``i`` between
  the ``i``-th hottest evaporating and condensing levels (the shorter list
  repeats its last level);
- cascade topology: ``n + m - 1`` stages, each lifting from the next level down
  to its own. A stage above another condenser stage evaporates
  ``HPR_DT_CASCADE_HX`` below that stage's level, and a stage below another
  evaporator stage condenses ``HPR_DT_CASCADE_HX`` above it, as in cascade
  targeting;
- cascade heat pumps also give vapour compression plus MVR: the MVR serves one
  or more of the hottest chosen condensing levels and parallel vapour
  compression the rest. The MVR starts ``HPR_DT_CASCADE_HX`` below the hottest
  vapour compression level and lifts in equal stages of at most 20 K. An MVR is
  an open cycle, so its stages screen water only.

A single-stage or cascade layout with a stage lifting more than
``HPR_SCREEN_MAX_STAGE_LIFT`` (70 K by default) is also screened as a cascade
that splits that stage into the fewest equal lifts within the limit, allowing
``HPR_DT_CASCADE_HX`` between them. The inserted levels are condensing levels,
except where the stage joins the evaporating side of a refrigeration cascade;
the layout lists them as ``added_levels``. Parallel machines and the vapour
compression stages under an MVR are not split, since no targeting method runs a
cascade inside them.

A layout with a stage that has no lift is dropped. Each stage screens every
candidate fluid between its own evaporating and condensing temperatures and
reports the critical temperature, the evaporating (dew point) and condensing
(bubble point) saturation pressures, their ratio, and the suction volumetric
heating capacity: the evaporating latent heat times the saturated vapour
density, in kJ/m3. The verdict is the first of these that applies:

1. ``above critical``: the condensing level is within ``HPR_DT_CONT`` of the
   critical temperature. The vapour-compression model is subcritical only.
2. ``below triple point``: the evaporating level is below the fluid's minimum
   temperature in CoolProp.
3. ``over ratio limit``: the pressure ratio is above
   ``HPR_MAX_PRESSURE_RATIO``.
4. ``deep vacuum suction``: the evaporating pressure is below
   ``HPR_SCREEN_MIN_SUCTION_PRESSURE`` (0.5 bar by default).
5. ``near critical``: the condensing level is within
   ``HPR_SCREEN_CRITICAL_MARGIN`` (5 K by default) of the critical temperature.
   With the margin no wider than ``HPR_DT_CONT``, no fluid is near critical.
6. ``practical``: none of the above.

The first four are hard limits: a fluid is feasible on a stage only when it is
practical or near critical. Fluids above critical or below their triple point
have no pressures or capacity. Within each stage the feasible fluids come
first, ranked by suction volumetric heating capacity; a larger capacity means a
smaller compressor for the same duty.

A fluid set takes one feasible fluid per vapour compression stage, hottest
stage first, and its critical temperature never rises down the stages: the
targets sort a refrigerant list that way, so a hotter stage never runs a fluid
with a lower critical temperature. The same fluid may repeat. Each layout
reports ``n_fluid_sets``, and ``fluid_sets()`` lists them; a layout whose MVR
stage is infeasible has none. Passing the screen as ``refrigerants`` to a vapour
compression target runs every fluid set of the layouts for that target's
method and reports the cheapest by total annualized cost.

The candidates default to every pure and pseudo-pure fluid in the installed
CoolProp, so the list can differ between CoolProp versions;
``refrigerants="natural"`` selects water, ammonia, CO2, propane, n-butane,
isobutane, n-pentane, isopentane and neopentane. Mixtures are not screened.

The screen is a first filter, not a cycle simulation: it ignores compressor
discharge temperature, superheat and subcooling, and the detailed search may
move the levels away from the Carnot ones.

Winning Design and Part-Load Maps
---------------------------------

A successful compatible single-stage target retains a frozen
``HprTargetSimulationRecord``. It identifies the selected backend, model,
working fluid, nominal temperatures and useful duty, approach temperatures,
cycle assumptions, engine version, and the compressor-only power boundary. It
contains plain values, not a CoolProp or TESPy object.

``problem.target.hpr_performance_map(target=..., request=...)`` extracts its
basis only from that winning record. With TESPy, targeting candidates use fresh
design solves; map generation prepares the winning design and then uses
offdesign solves at the requested source temperature, sink temperature, and
positive load-fraction grid. Backend identity is target-owned, so changing
current problem configuration cannot silently switch the map engine.

Map generation is atomic: one invalid or non-converged requested point returns
no partial map. There is no automatic temperature grid, zero-load point, unit
commitment, capacity decision, or time coupling. Schema ``1.0`` represents a
fixed reference unit. A downstream optimizer may scale ``q_source``,
``q_sink``, and ``electric_power`` together and interpolate only between
adjacent points on each ordered part-load curve.

OpenUtility is such a downstream consumer. It reads the versioned plain mapping
and owns Pyomo, HiGHS, electricity balances, thermal balances, unit commitment,
and multiperiod dispatch. OpenPinch does not import OpenUtility, and OpenUtility
does not need to import OpenPinch to validate or optimize exported JSON.

Multiperiod Targeting Modes
---------------------------

``problem.target.all_periods.<method>(...)`` runs an independent target for
each period and returns ordered period results. It does not optimize one
installed design across those periods.

For one shared installed design, call the scalar Carnot, CoolProp
vapour-compression, or VC+MVR method with
``options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True}``. The selected
``period_id`` is the reporting period while the optimizer evaluates the full
problem period set. A successful shared result exposes ``design_vector``,
``period_ids``, ``period_weights``, ``period_outputs``, and ``weighted_output``
through ``hpr_details``. Verify the exact period set, aligned weights, finite
design and objective values, and successful output for every period.

Use a fresh ``PinchProblem`` for each technology comparison. Shared-design
Carnot heat pumping, Carnot refrigeration, CoolProp heat pumping, CoolProp
refrigeration, and CoolProp VC+MVR are separate optimizations rather than one
joint technology-selection model. Shared-vector multiperiod TESPy targeting is
not supported.

Interpretation
--------------

Compare candidate results using utility reduction, recovered heat, COP, work,
temperature lift, and annualized cost. A thermodynamically feasible candidate
is not automatically the lowest-cost retrofit; interpret HPR targets alongside
the Grand Composite Curve and net-load profiles.

Simulated-cycle integration accounting ranks targets by annualized cost:
electricity, external heating, cooling water and default refrigeration, the
heat pump's installed capital, and the installed capital of the default
utilities the design still needs. Heat-pump capital scales with heating capacity to the power 0.7, with
the number of compression stages and with the delivery temperature above
75 °C; its defaults are fitted to the IEA HPT Project 68 cost data. Heat pumps
do not always pay, particularly at high lift. Multiperiod summaries weight
operating quantities and size shared equipment, and the default-utility
capacity, against the governing period rather than averaging away the peak
design requirement. See :doc:`../guides/heat-pump-workflows` for the cost
equation and its options.

See :doc:`../guides/heat-pump-workflows` and notebooks 08 through 11 in
:doc:`../examples/notebook-series`.
