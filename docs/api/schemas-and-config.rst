Schemas and Config
==================

OpenPinch has two distinct but closely related internal typed surfaces:

- schema models for external inputs and returned results
- a runtime :class:`~OpenPinch.domain.configuration.Configuration` object attached to
  each prepared zone

Together they implement the transport format used by
:class:`OpenPinch.PinchProblem` and the per-zone analysis behaviour.

What Each Layer Does
--------------------

``TargetInput`` and related schemas
   Define the request format for process streams, utilities, flat
   ``options``, the optional zone tree, an optional serialized heat exchanger
   network, saved ``process_components``, and the ``residual_basis`` and
   ``plant_profile_data`` fields.

``TargetOutput`` and target/result schemas
   Define the structured response returned by the top-level service boundary.

``Configuration``
   Stores numerical and engineering defaults for input and output units,
   thermal and direct-integration settings, costing, heat pumps, ORC,
   turbines, process MVR and heat exchanger network synthesis. Its method selectors are not user-settable.
   Each prepared :class:`~OpenPinch.domain.zone.Zone` owns one config object.

Discovering Options
-------------------

Use ``configuration_options()`` to inspect the
supported flat ``TargetInput.options`` keys, their groups, runtime status,
enum choices, numeric bounds, and config paths:

.. code-block:: python

   from OpenPinch.presentation.configuration import configuration_options

   options = configuration_options()
   hpr_options = [field for field in options if field.group == "hpr"]

The same metadata is tabulated on :doc:`configuration-options`.

Common options include ``THERMAL_DT_CONT`` for minimum contribution
temperature, ``OUTPUT_UNIT_*`` fields for report units, and numerical
``HPR_*`` fields for heat-pump and refrigeration workflows.

``THERMAL_DT_CONT`` is the ``dt_cont`` of any process stream or utility that
doesn't give its own (an absent, null or blank value). Heat exchanger network
synthesis scales every contribution by each ``HENS_DT_CONT_MULTIPLIERS``
value, as a zone ``dt_cont`` multiplier does in targeting, so the default
``[1.0]`` designs at the targeting contributions. A utility without a ``price`` uses
``COSTING_UTILITY_PRICE``, the price of the default utilities.

These defaults, ``THERMAL_DT_PHASE_CHANGE``, ``THERMAL_HTC``, the
``INPUT_UNIT_*`` fields and the ``PROBLEM_*`` fields are applied when the
problem is loaded. Change them with ``update_options(...)`` or a workspace
scenario; passing them in a single ``target.*`` or ``design.*`` call's
``options`` raises a ``ValueError``.

A negative ``dt_cont`` is allowed, with a validation warning, as a deliberate
modelling choice. It lets the real composite curves cross, so the degree of
integration can exceed 1, and area targeting reports infinite area and capital
cost (with a warning) instead of failing. A zero approach (``THERMAL_DT_CONT``
of 0 or a zero ``dt_cont_multiplier``) gives infinite area in the same way.
Streams whose supply and target temperatures differ by less than
``THERMAL_DT_PHASE_CHANGE`` are widened to that span, with the same duty, so
they are never lost between temperature intervals.

Streams that share a name within a zone are renamed with a numeric suffix
(``_1``, ``_2``). This is reported as a validation warning only when
``REPORTING_DEBUG_ENABLED`` is true; it is off by default because published
data often gives one stream as several same-named segments.

Target-method selectors such as ``HPR_TYPE``, ``ORC_MODEL`` and
``POWER_TURB_MODEL`` are not user-settable: passing one in ``options`` raises a
``ValueError``. The descriptive ``problem.target.*`` or ``problem.design.*``
callable selects the analysis and sets the selector for that run.

Configuration
-------------

.. autoclass:: OpenPinch.domain.configuration.Configuration
   :members:
   :no-index:

Input and Output Schemas
------------------------

.. autoclass:: OpenPinch.contracts.input.TargetInput
   :members:
   :no-index:

``TargetInput.network`` accepts the mapping emitted by a runtime
:class:`~OpenPinch.domain.heat_exchanger_network.HeatExchangerNetwork`, while
remaining an independent transport schema:

.. code-block:: python

   from OpenPinch.contracts.input import TargetInput

   network_payload = network.model_dump(mode="json")
   input_data = TargetInput.model_validate(
       {
           "streams": [],
           "utilities": [],
           "network": network_payload,
       }
   )

   assert input_data.model_dump(mode="json")["network"] == network_payload

Use ``model_dump(mode="json")`` for this bridge. ``model_dump_json()`` returns
an encoded string and must be decoded before it can be supplied as the nested
``network`` value. The network is retained in canonical input data, but it is
not automatically consumed as a synthesis seed. Endpoint classifications use
the exact :class:`~OpenPinch.domain.enums.StreamID` values ``Process`` and
``Utility``; lowercase legacy values and ``Unassigned`` are rejected.

.. autoclass:: OpenPinch.contracts.input.HeatExchangerNetworkSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.HeatExchangerSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.HeatExchangerPeriodStateSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.HeatExchangerAreaSliceSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.output.TargetOutput
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.StreamSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.StreamSegmentSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.TemperatureHeatPointSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.TemperatureHeatProfileSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.UtilitySchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.input.ZoneTreeSchema
   :members:
   :no-index:

Heat Exchanger Network Design Results
-------------------------------------

Every ``problem.design.*`` call returns an explicit design view. Inspect its
selected network with ``design.selected_network``, choose a ranked candidate
with ``design.network(rank=...)``, list candidates with ``design.top(n)``, or
render one with ``design.grid(rank=...)``. Convenience totals and
``design.utility(name)`` always refer to ``selected_network``.

The complete serializable synthesis result is ``design.result`` and is also
stored at ``problem.results.design``. Serialize it explicitly:

.. code-block:: python

   design = problem.design.heat_exchanger_network()
   payload = design.result.model_dump(mode="json")

The result's ``ranked_networks``, ``manifest``, ``diagnostic_references``,
task metadata, ``design_method``, and task ``method`` remain available through
``design.result``. The design view does not forward unknown attributes and is
not itself a Pydantic model.

The
:class:`~OpenPinch.domain.enums.HeatExchangerNetworkDesignMethod` enum is the
single method identity used for dispatch and result metadata.
``design.result.manifest.method_sequence`` records the executed task sequence:
a stage that the tier leaves out, or that could not run because Couenne is
missing, is not listed.
For tiered OpenHENS runs, the manifest also records the synthesis quality tier,
selected pathway, and protected-fallback status.

``HENS_SYNTHESIS_QUALITY_TIER`` remains a persistent configuration field with a
default of tier 1 for prepared-problem workflows. If the problem's options set
it, ``problem.design.heat_exchanger_network()``, its multiperiod form and
``problem.design.enhanced_heat_exchanger_network()`` run that tier. If they do
not, those methods run their own defaults, tiers 0 and 2, even though the
configuration still reads 1; the key in ``problem.to_problem_json()["options"]``
is what counts as set. ``problem.design.open_hens()`` always runs tier 1. The
tier that ran is in ``design.result.manifest.synthesis_quality_tier`` and in the
``HENS_SYNTHESIS_QUALITY_TIER`` value of ``design.result.provenance``. User
code should prefer
``problem.design.enhanced_heat_exchanger_network(quality_tier=...)`` for method-level
tier selection because it applies a call-local override without mutating the
loaded problem configuration. That argument wins over a tier in the call's
``options``, which wins over the stored option for that call only. Runtime
``options`` passed to design accessors accept other ``HENS_*`` overrides the same
way: they apply to that call only and do not change the stored configuration.

``HENS_MIN_EXCHANGER_APPROACH_SHARE`` (default 0, at least 0 and below 1) sets
the smallest exchanger approach the thermal derivative and network evolution
steps may choose, as a share of the requested approach temperature. At the
default those steps choose approaches by cost down to 0.1 K; see
:doc:`../guides/heat-exchanger-network-synthesis`.

``HENS_EVM_N_AD_BRANCHES`` and ``HENS_EVM_N_RM_BRANCHES`` set the network evolution
breadth of every route when they are set, including the tier 0 and tier 1 routes that
tiers 2 to 5 run as protected routes, and including ``problem.design.open_hens()``.
With either option above 1, the tier 1 route is no longer the original search. When
they are not set, tier 0 and tier 1 routes use a breadth of 1 in any tier, while tiers
4 and 5 use 2 on their expanded routes. ``manifest.evm_n_ad_branches`` and
``manifest.evm_n_rm_branches`` report the widest of these, so at tier 4 or 5 a
protected route can be selected at a narrower breadth than the manifest shows: check
``manifest.selected_protected_pathway``.

Method-level inputs and outputs are Pydantic models. Their shared input
contract contains run/problem metadata, settings, optional seed network,
optional seed-network index, and trace metadata. Their shared output contract
contains status, accepted networks, ranked networks, diagnostics, trace
metadata, and an optional manifest.

Ranks passed to ``design.network(...)`` and ``design.grid(...)`` are one-based.

Grid diagrams for the selected network are created with
:func:`OpenPinch.presentation.network_grid.service.build_grid_diagram`. The
service accepts one or more
:class:`~OpenPinch.domain.heat_exchanger_network.HeatExchangerNetwork`
objects. Select a ranked network first when needed. Multiperiod networks
require an explicit period for duties, temperatures, diagrams, exports, and
controllability; omission is accepted only for a single-period network. The
returned object wraps the
Plotly ``fig``, a lightweight drawing adapter ``ax``, the selected ``network``,
and the normalized ``grid_model`` used to draw the topology.

For clients that draw the network themselves, the design view's
``view(rank=, period_id=)`` returns a
:class:`~OpenPinch.contracts.synthesis.view.HeatExchangerNetworkView`: the
duplicate-free ranking table, one ranked network's duty totals and utility
duties, and its grid layout
(:class:`~OpenPinch.contracts.synthesis.view.HeatExchangerNetworkGridLayout`,
also from ``grid_layout(rank=, period_id=)``) as plain data: stream rows,
stages, matches with their duty, temperatures and area, branch counts, and
each stream's supply and target temperature in the period
(``stream_temperatures``), in the network's units. A shared
multiperiod network needs ``period_id``; ``views(rank=)`` returns one view per
period, each naming its ``period_id``.

.. autoclass:: OpenPinch.contracts.synthesis.result.HeatExchangerNetworkSynthesisResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.synthesis.view.HeatExchangerNetworkView
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.synthesis.view.HeatExchangerNetworkGridLayout
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.synthesis.method.HeatExchangerNetworkSynthesisMethodInput
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.synthesis.method.HeatExchangerNetworkSynthesisMethodOutput
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.enums.HeatExchangerNetworkDesignMethod
   :members:
   :no-index:

Target Models
-------------

Solved targets are normalized through the target schema layer before they are
returned to users or exported.

.. autoclass:: OpenPinch.domain.targets.BaseTargetModel
   :members:
   :no-index:

HPR Schemas
-----------

The HPR schema layer carries the prepared configuration values, parsed backend
state, and simulated-cycle annualized cost accounting used by the targeting
services. Report-facing HPR cost fields use ``Value`` instances with serialized
units ``$`` and ``$/y``. Internal parsed-state and backend-result records are
attribute-only; call ``model_dump()`` only when mapping data is required. HPR
optimiser configuration accepts the exact identifiers ``dual_annealing``,
``cmaes``, ``bo``, and ``rbf_surrogate``.

.. autoclass:: OpenPinch.contracts.hpr.HPRParsedState
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.HeatPumpTargetInputs
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.HPRBackendResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.SimulatedHPRAnnualizedCostAccounting
   :members:
   :no-index:

HPR Performance-Map Contract
----------------------------

The specialist performance-map contract is a plain-data boundary for exporting
active heat-pump and refrigeration operating points. Schema version ``1.0`` is
closed: unknown fields and unknown versions are rejected rather than migrated.
It is deliberately separate from the runtime HPR target records above.

Compatible successful scalar targets retain one detached nominal record used
to construct a map basis without retaining a thermodynamic engine object:

.. autoclass:: OpenPinch.contracts.hpr.HprTargetSimulationRecord
   :members:
   :no-index:

``HprPerformanceMap`` contains exactly ``schema_version``, ``map_id``, ``mode``,
``units``, ``reference_capacity``, ``reference_capacity_basis``,
``interpolation_topology``, ``thermodynamic_backend``, ``model_id``,
``provenance``, ``points``, ``cop_convention``,
``energy_balance_tolerance``, and ``temperature_match_tolerance``. Each
``HprPerformancePoint`` contains ``name``, ``curve_id``, source and sink
temperatures, ``load_fraction``, ``q_source``, ``q_sink``, ``electric_power``,
and ``cop``.

Validation enforces canonical ordering, unique names and coordinates, one
source/sink temperature pair per ordered part-load curve, physical energy and
COP identities, and the mode-specific capacity basis. The models are immutable
after validation. Structured ``provenance`` is restricted to JSON-compatible
values so a dumped map remains independent of OpenPinch Python objects.

.. autoclass:: OpenPinch.contracts.hpr_performance_map.HprPerformanceMapRequest
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr_performance_map.HprPerformanceMapUnits
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr_performance_map.HprPerformancePoint
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr_performance_map.HprPerformanceMap
   :members:
   :no-index:

Enums and Typed Constants
-------------------------

The :mod:`OpenPinch.domain.enums` module owns stream types, target labels, HPR
cycle selectors, turbine model choices, and other canonical identifiers.

.. automodule:: OpenPinch.domain.enums
   :members:
   :no-index:

Design Notes
------------

The schema layer should be the source of truth for external input contracts.
The configuration layer should be the source of truth for runtime toggles and
per-zone behavior. Keeping those roles distinct is what makes the package
predictable when used from notebooks and services.
