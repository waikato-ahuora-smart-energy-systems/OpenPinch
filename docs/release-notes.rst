Pre-Release Notes
=================

Unreleased
----------

Result fixes
~~~~~~~~~~~~

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
