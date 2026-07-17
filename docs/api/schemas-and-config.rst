Schemas and Config
==================

OpenPinch has two distinct but closely related internal typed surfaces:

- schema models for external inputs and returned results
- a runtime :class:`~OpenPinch.domain.configuration.Configuration` object attached to
  each prepared zone

Together they implement the wire format protected through
:func:`OpenPinch.main.pinch_analysis_service` and the per-zone analysis
behaviour. Their direct import paths are not compatibility contracts.

What Each Layer Does
--------------------

``TargetInput`` and related schemas
   Define the request format for process streams, utilities, and the
   optional zone tree.

``TargetOutput`` and target/result schemas
   Define the structured response returned by the top-level service boundary.

``Configuration``
   Stores runtime knobs for targeting flags, heat pump parameters, utility
   assumptions, costing inputs, and turbine settings. Each prepared
   :class:`~OpenPinch.domain.zone.Zone` owns one config object.

Discovering Options
-------------------

Use ``config_options()`` or ``Configuration.options_catalog()`` to inspect the
supported flat ``TargetInput.options`` keys, their groups, runtime status,
enum choices, numeric bounds, and config paths:

.. code-block:: python

   from OpenPinch.presentation.configuration import configuration_options as config_options

   options = config_options()
   hpr_options = [field for field in options if field.group == "hpr"]

Common options include ``THERMAL_DT_CONT`` for minimum contribution
temperature, ``TARGETING_*`` flags for default target selection,
``OUTPUT_UNIT_*`` fields for report units, and ``HPR_*`` fields for heat-pump
and refrigeration workflows.

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

``TargetOutput.design`` stores a
:class:`~OpenPinch.contracts.synthesis.result.HeatExchangerNetworkSynthesisResult`
after ``problem.design.enhanced_synthesis_method(quality_tier=...)``,
``problem.design.open_hens_method()``,
``problem.design.heat_exchanger_network_synthesis()``, or one of the direct
method accessors runs. The selected network is available as ``design.network``
and the ranked unique network candidates are available as
``design.ranked_networks``.

The same
:class:`~OpenPinch.domain.enums.HeatExchangerNetworkDesignMethod` enum is used for
internal dispatch and task/result method metadata. ``HENDesignMethod`` is the
short internal alias. ``design.design_method`` records the requested design
service that was requested, such as ``HENDesignMethod.OpenHENS``. ``design.method``
records the task method that produced the selected network, such as
``HENDesignMethod.NetworkEvolution`` for a normal OpenHENS sequence result.
``design.manifest.method_sequence`` records the executed task-level method
sequence. For OpenHENS tiered runs, ``design.manifest.synthesis_quality_tier``
records the call-local or configured tier, ``selected_pathway_id`` records the
winning pathway, and ``selected_protected_pathway`` indicates whether the
selected network came from a protected fallback route.

``HENS_SYNTHESIS_QUALITY_TIER`` remains a persistent configuration field with a
default of tier 1 for prepared-problem workflows. User code should prefer
``problem.design.enhanced_synthesis_method(quality_tier=...)`` for method-level
tier selection because it applies a call-local override without mutating the
loaded problem configuration. Runtime ``options`` passed to design accessors are
reserved for runtime context and do not accept persistent ``HENS_*`` overrides.

Method-level inputs and outputs are also Pydantic models. Their shared input
contract contains run/problem metadata, settings, optional seed network,
optional seed-network index, and trace metadata. Their shared output contract
contains status, accepted networks, ranked networks, diagnostics, trace
metadata, and an optional manifest.

Use ``design.get_n_best_networks(n)`` to read the first ``n`` ranked
candidates. Use ``design.select_network(solution_rank=...)`` to make another
ranked candidate the selected ``design.network``. ``solution_rank`` is 1-based.

The problem-level design accessor exposes convenience totals for the selected
network at ``problem.design.network``:
``total_heat_recovery``, ``total_hot_utility``, ``total_cold_utility``, and
``utility(name)``.

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

.. autoclass:: OpenPinch.contracts.synthesis.result.HeatExchangerNetworkSynthesisResult
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
predictable when used from notebooks, services, and the CLI.
