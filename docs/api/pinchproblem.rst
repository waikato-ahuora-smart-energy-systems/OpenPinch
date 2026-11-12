PinchProblem
============

:class:`OpenPinch.PinchProblem` is the canonical stateful workflow for one
process-engineering case.

Lifecycle
---------

``prepared``
   Construction or ``load(...)`` validates and prepares streams, utilities,
   zones, periods, and configuration. Input replacement is atomic: if loading,
   validation, or preparation fails, the prior prepared problem and cached
   results remain available together with their original input.

``targeted``
   A descriptive ``target`` method stores the latest result. All-period methods
   additionally populate the ordered ``period_results`` cache.

``designed``
   A descriptive ``design`` method returns a HEN design view with ranked
   network selection and grid rendering.

``invalidated``
   Loading new input, changing stored options, changing the temperature-approach
   contribution, or mutating
   a process component clears results that no longer describe the prepared
   problem.

Interaction Matrix
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 27 18 18 13

   * - Surface
     - Purpose
     - Return
     - State effect
     - Dependency
   * - ``load``, ``validate``, ``validation_report``, ``to_problem_json``
     - Prepare, check, and serialize input
     - zone, validated input, report, or mapping
     - prepare or observe
     - base
   * - ``source_validation_report`` (classmethod)
     - Report on a source that may be invalid, without raising
     - report
     - none; no problem is kept
     - base
   * - ``target.direct_heat_integration``, ``indirect_heat_integration``,
       ``total_site_heat_integration``, ``all_heat_integration``
     - Core Pinch and Total Site analysis
     - target output
     - targeted
     - base
   * - ``target.heat_exchanger_area_and_cost``, ``exergy``,
       ``energy_transfer``
     - Enrich a thermal target
     - target output
     - targeted
     - base
   * - ``target.heat_recovery_dt_min``
     - Invert requested process recovery to an equivalent global ``dt_min``
     - frozen diagnostic result
     - source unchanged
     - base
   * - ``target.hpr_load_kinks``
     - List the net load profile vertices and the logical HPR load fractions
     - frozen diagnostic result
     - source unchanged
     - base
   * - ``target.carnot_*``, ``vapour_compression_*``, ``mvr_heat_pump``
     - Model-specific HPR studies. ``brayton_*`` raises
       ``NotImplementedError`` pending solver repair
     - target output
     - targeted
     - base; TESPy for ``simulation_backend="tespy"``
   * - ``target.hpr_refrigerant_screen``
     - Screen fluids on every parallel or cascade layout of a Carnot target's
       saturation levels or explicit levels
     - frozen screening result
     - source unchanged
     - base
   * - ``target.cogeneration`` and named turbine-model methods
     - Cogeneration screening
     - target output
     - targeted
     - base
   * - ``target.organic_rankine_cycle``, ``carnot_orc``
     - Organic Rankine cycle screening
     - target output, or ``None`` when there is no surplus below the pinch
     - targeted
     - base
   * - ``target.utility_placement``
     - Optimize or infer hot and cold utility levels at a selected hierarchy zone
     - solved detached optimized problem
     - source unchanged
     - base
   * - ``target.all_periods.*``
     - Mirror supported targeting over ordered periods; ``utility_placement``
       forwards to the single detached placement call
     - period-to-output mapping
     - period cache
     - method-specific

   * - ``components.add_process_mvr``, ``components.inventory``
     - Add or inspect process MVR mutations
     - component or mapping
     - invalidates on mutation
     - base
   * - ``design.*heat_exchanger_network``, ``open_hens``, ``pinch_design``,
       ``thermal_derivative``, ``network_evolution``, ``optimise_duties``
     - HEN synthesis and improvement
     - design view
     - designed
     - HEN solver
   * - ``summary_frame``, ``metrics``, ``report`` and state properties
     - Inspect prepared or cached state
     - dataframe, mapping, report, or record
     - none
     - base
   * - ``compare_to``
     - Compare the numeric summary metrics of two solved problems
     - dataframe
     - none
     - base
   * - ``update_options``, ``set_dt_cont_multiplier``
     - Change stored options or the temperature-approach contribution
     - zone
     - re-prepare; clears stale results
     - base
   * - ``residual_utility``, ``with_utilities_from``
     - Derive a new unsolved problem from a residual HPR target or a donor's
       utilities
     - new problem
     - source unchanged
     - base
   * - ``plot.catalog``, ``plot.data``, and named plot methods
     - Inspect cached graph data or build a figure
     - catalog, mapping, or figure
     - none
     - plotting
   * - ``plot.export``, ``plot.export_gallery``, ``export_excel``,
       ``show_dashboard``
     - Explicit publication side effects
     - paths; ``show_dashboard`` returns none
     - none
     - output-specific

Inverse Heat-Recovery Target
----------------------------

Use the inverse target when process heat recovery is specified and the
corresponding global ``dt_min`` is unknown:

.. code-block:: python

   problem = PinchProblem("basic_pinch.json")
   result = problem.target.heat_recovery_dt_min(
       heat_recovery={"value": 4_000.0, "unit": "kW"},
       zone="Site/Plant",
       period_id="0",
   )

   payload = result.model_dump(mode="json")

Plain recovery values use the configured input heat-flow unit; scalar
``Value``/Pint quantities and exact ``{"value", "unit"}`` mappings are also
accepted. Input shapes are strict: Booleans, numeric strings, sequences,
arrays, and unrelated mappings are rejected. The result reports the requested and achieved
recovery, zero-``dt_min`` thermodynamic limit, residual, status, and iteration
count with explicit output units. Requests above the thermodynamic limit are
rejected with the request, limit, scope, period, and units in the error.

``HeatRecoveryDtMinResult`` is frozen and contains ``scope``, canonical
``period_id``, ``dt_min``, ``requested_heat_recovery``,
``achieved_heat_recovery``, ``thermodynamic_limit``,
``heat_recovery_residual``, ``status``, ``iterations``, and ``provenance``. Status is
``solved`` for an interior request, ``at_thermodynamic_limit`` for the
greatest approach retaining maximum recovery, or ``zero_recovery_boundary``
for the first approach that produces zero recovery. The
``at_thermodynamic_limit`` approach may be positive for a threshold problem;
zero approach defines how the maximum recovery is calculated, not a mandatory
inverse result.

For all periods, a scalar broadcasts deliberately, while a mapping must contain
exactly the canonical period IDs. Exact period keys take precedence when the
IDs themselves are ``value`` and ``unit``:

.. code-block:: python

   problem = PinchProblem("crude_preheat_train_multiperiod.json")
   period_results = (
       problem.target.all_periods.heat_recovery_dt_min(
           heat_recovery={"base": 4_000.0, "turndown": 3_200.0, "peak": 4_800.0},
           workers=2,
       )
   )

The inverse service is non-mutating: it does not replace ordinary target
results, populate ``period_results``, alter stream contributions, or update the
last target-run specification. The value is process-level global ``dt_min``,
not exchanger-level EMAT.

Supported scopes are Site, Process Zone, and Unit Operation. Community and
Region scopes are rejected because they are not direct process-targeting
scopes. A ``Zone`` argument contributes only its address, which is resolved
against the current problem or current batch case. On an interior plateau, the returned value is the greatest feasible
approach that still meets the requested recovery.

.. autoclass:: OpenPinch.contracts.heat_recovery_dt_min.HeatRecoveryDtMinResult
   :members:
   :no-index:

The complete runnable workflow, validation matrix, unit behavior, workspace
batch examples, and numerical interpretation are in
:doc:`../guides/heat-recovery-dt-min`.

Refrigerant Screening
---------------------

``problem.target.hpr_refrigerant_screen`` screens candidate working fluids on
every layout of the evaporating and condensing levels of a successful Carnot
heat pump or refrigeration target: single stage, parallel or cascade (as the
target's topology), and vapour compression plus MVR for a cascade heat pump.
The target must be the problem's current one: a target from another problem,
or one superseded or made stale by later changes, is rejected. Pass the target,
or both ``T_evap`` and ``T_cond`` (one level or a list; degC or
``{"value", "unit"}``) with ``topology`` (``"parallel"`` or ``"cascade"``) and,
for refrigeration, ``is_refrigeration=True``. The result has one layout per
method and choice of levels, each with its stages, ranked fluids and fluid
sets. It does not change the problem.

.. code-block:: python

   heat_pump = problem.target.carnot_heat_pump(load_fraction=0.25)
   screen = problem.target.hpr_refrigerant_screen(
       target=heat_pump, refrigerants="natural"
   )
   for layout in screen.feasible_layouts:
       layout.method, layout.n_fluid_sets, layout.fluid_sets()

   problem.target.hpr_refrigerant_screen(
       T_evap=50.0, T_cond=[180.0, 120.0, 90.0], topology="cascade"
   )

``refrigerants`` defaults to every fluid in the installed CoolProp; pass
``"natural"`` for water, ammonia, CO2 and the hydrocarbons, or a list of fluid
names. See :doc:`../fundamentals/heat-pump-and-refrigeration-methods` for the
verdicts and thresholds.

.. autoclass:: OpenPinch.contracts.hpr_refrigerant_screen.HprRefrigerantScreenResult
   :members:
   :no-index:

The vapour compression targets (``vapour_compression_heat_pump``,
``vapour_compression_refrigeration`` and ``mvr_heat_pump``) take the screen, or
one of its layouts, as ``refrigerants`` and run once for every fluid set it
proposes for their method: single-stage and cascade layouts for a cascade
cycle, single-stage and parallel layouts for a parallel cycle, and vapour
compression plus MVR layouts for ``mvr_heat_pump``. Each run takes its condensers, evaporators and
MVR stages from its layout, so a split layout runs one condenser on each of its
``added_levels``; pass ``condensers`` or ``evaporators`` to run only the layouts
with those counts. Layouts asking for the same counts and fluids
share one run. A run that fails is kept with its error, and ``best`` is the
successful run with the lowest total annualized cost. One fluid set from
``fluid_sets()``, like a refrigerant list, runs once and returns the target.

.. code-block:: python

   runs = problem.target.vapour_compression_heat_pump(
       refrigerants=screen, load_fraction=0.25
   )
   for run in runs.runs:
       run.fluids, run.success, run.total_annualized_cost, run.error
   best = runs.best.target

Every run commits like a single target call, so afterwards the problem holds
the last successful run; each run's ``target`` can still be passed on as
``target=``.

.. autoclass:: OpenPinch.contracts.hpr_fluid_set_runs.HprFluidSetRunsResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr_fluid_set_runs.HprFluidSetRun
   :no-index:

HPR Load Kinks
--------------

``hpr_load_kinks`` finds the logical heat pump and refrigeration loads from
kinks in the net load profiles of the direct-integration problem table. It runs
no heat pump and changes nothing in the problem:

.. code-block:: python

   problem = PinchProblem("basic_pinch.json")
   kinks = problem.target.hpr_load_kinks()
   kinks.logical_load_fractions  # (0.4..., 1.0)
   heat_pump = problem.target.carnot_heat_pump(
       load_fraction=kinks.logical_load_fractions[0]
   )

A heat pump walks the sink profile (``H_NET_COLD``) up from the pinch;
``is_refrigeration=True`` walks the source profile (``H_NET_HOT``) down from it.
The slope at each vertex is the shifted-temperature change per unit load
fraction. Each vertex has a ``kind``:

- ``step``: the profile changes temperature with no duty after the vertex.
- ``steepens``: the slope after is at least ``steepening`` (default 2.0) times
  the slope before.
- ``flattens``: the slope before is at least ``steepening`` times the slope
  after.
- ``minor``: any smaller change.

A heat pump sized up to a step or steepening kink condenses below it; one sized
past it condenses across the step. ``logical_load_fractions`` lists the step and
steepens vertices in profile order, followed by 1.0. The fractions are exact,
not rounded, so passing one as ``load_fraction`` sizes the target to the kink.

``HprLoadKinksResult`` is frozen and contains ``scope``, canonical
``period_id``, ``is_refrigeration``, ``steepening``, ``maximum_duty``,
``vertices`` (each with ``load_fraction``, ``duty``, ``temperature``,
``slope_before``, ``slope_after`` and ``kind``; a step has no ``slope_after``),
``logical_load_fractions`` and ``provenance``. Duties and temperatures follow
the configured output units; slopes are temperature differences.
``problem.target.all_periods.hpr_load_kinks(workers=...)`` returns one result
per canonical period. Supported scopes are Site, Process Zone and Unit
Operation. A profile with no duty, or ``steepening`` of 1 or less, raises
``ValueError``.

.. autoclass:: OpenPinch.contracts.hpr_load_kinks.HprLoadKinksResult
   :members:
   :no-index:

Argument Precedence
-------------------

Effective arguments resolve as ``named keyword > options > stored config >
default``. Named keywords and ``options`` apply only to that call. Use
``update_options(...)`` when a later call should inherit a persistent
engineering value. Target and design method selectors are not user-settable
options; the ``target`` or ``design`` method you call selects the method.

Process MVR Component Results
-----------------------------

``components.add_process_mvr(...)`` returns the component it created. Use
engineering argument names such as ``compressor_efficiency`` and
``motor_efficiency``. The returned object exposes ``active``, ``activate()``,
``deactivate()``, ``original_streams``, ``replacement_streams``,
``stage_results_by_period``, ``affected_zone_paths``, and ``work_for_zone()``.
It raises ``DirectGasMVRPhaseError`` (a ``ValueError``) if a stage would cross
into liquid, naming the stage, temperatures and pressure; see
:doc:`../reference/api-heat-pump`.
Changing component activity invalidates cached targets, so rerun the chosen
target method afterward.

Class Members
-------------

The members below are those defined on the class itself. The ``target``,
``design``, ``plot`` and ``components`` accessors are implemented by internal
classes and are not expanded here, so their full signatures and defaults are
not listed on this page.

.. autoclass:: OpenPinch.PinchProblem
   :members:
   :undoc-members:

The operation-level inventory and tutorial owner for every member is published
in :doc:`../examples/tutorial-coverage-map`.

Derived cases and explicit bases
--------------------------------

``problem.residual_utility(base_target=heat_pump)`` creates an unsolved case
retaining the frozen HPR residual. ``problem.with_utilities_from(other_problem,
project_name=None)`` creates an unsolved receiver-based study with the donor's
canonical utilities. Matching period sets are required and arrays follow period
identity; runtime components and receiver thermal inputs are preserved.

The keyword-only ``base_target=None`` on direct/all integration and utility
placement selects a compatible scalar HPR residual. Without it, the established
ordinary process workflow applies. With it, the source study remains unchanged:
direct integration returns ``ResidualUtilityTarget``, all integration returns
``TargetOutput``, and placement returns a solved residual ``PinchProblem``.
For all integration, omitted ``include_subzones`` means true for ordinary cases
and false for a scalar HPR or residual case; explicit expansion is unsupported
for a frozen scalar basis.

Exergy, energy transfer and the five cogeneration methods retain their existing
base-target capabilities and infer omitted scalar zone/period selections.
Area/cost, new HPR targeting and aggregate integration do not gain new basis
capabilities in this release. ``hpr_performance_map`` keeps its required
``target=`` argument, and ``hpr_refrigerant_screen`` takes its Carnot target as
``target=``. See :doc:`../guides/heat-pump-workflows` for examples,
unsupported selections and ownership rules.

Input JSON remains input-only. Reconstructing a solved placement case from
``to_problem_json()`` starts an unsolved study. Utility transfer does not copy
results, placement evidence or a donor's residual basis into the receiver.
