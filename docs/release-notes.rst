Pre-Release Notes
=================

Unreleased
----------

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

HPR cost model and search
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
