Heat Pump and Refrigeration Methods
===================================

OpenPinch separates engineering model choice into descriptive methods:

``carnot_heat_pump`` and ``carnot_refrigeration``
   Fast thermodynamic screening. Use utility-placement and cascade-topology
   booleans to express the physical arrangement.

``vapour_compression_heat_pump`` and ``vapour_compression_refrigeration``
   Refrigerant-specific simulated cycles with explicit refrigerant candidates.

``brayton_heat_pump`` and ``brayton_refrigeration``
   Brayton-cycle studies using the model-specific backend.

``mvr_heat_pump``
   Vapour-compression/MVR cascade targeting.

The method name selects the model; configuration does not. Named
``load_fraction``, ``load_duty``, or ``period_loads`` values avoid a separate
load-mode string. Only one load form may be supplied per call.

Placement and Topology
----------------------

Direct placement couples the HPR model to a process heat-integration target.
Utility placement couples it to the site utility system. For Carnot methods,
``is_utility_heat_pump`` or ``is_utility_refrigeration`` expresses that choice.

Parallel topology represents independent lifts. Cascade topology connects
successive temperature lifts and is selected with ``is_cascade_cycle`` where
the model supports it. MVR is a separate named workflow rather than a hidden
cycle-string combination.

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
zeotropic blends. A fluid unsupported at either state fails preflight before
optimization. The external ``REFPROP`` backend is outside this contract.

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

Simulated-cycle integration accounting retains the model-specific compressor,
expander, heat-exchanger, operating-cost, and annualized-capital contributions
used by result summaries. Multiperiod summaries weight operating quantities and
size shared equipment against the governing period rather than averaging away
the peak design requirement.

See :doc:`../guides/heat-pump-workflows` and notebooks 08 through 11 in
:doc:`../examples/notebook-series`.
