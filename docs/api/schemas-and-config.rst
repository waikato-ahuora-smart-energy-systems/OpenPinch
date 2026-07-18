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
   Define the request format for process streams, utilities, the optional
   zone tree, and an optional serialized heat exchanger network.

``TargetOutput`` and target/result schemas
   Define the structured response returned by the top-level service boundary.

``Configuration``
   Stores numerical and engineering defaults for heat pumps, utilities,
   costing, turbines, and solvers. It does not select core methods. Each prepared
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
temperature, ``OUTPUT_UNIT_*`` fields for report units, and numerical
``HPR_*`` fields for heat-pump and refrigeration workflows. Configuration does
not contain target-method selectors; the descriptive ``problem.target.*`` or
``problem.design.*`` callable selects the analysis.

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

The result's ``ranked_networks``, ``manifest``, diagnostics, task metadata,
``design_method``, and task ``method`` remain available through
``design.result``. The design view does not forward unknown attributes and is
not itself a Pydantic model.

The
:class:`~OpenPinch.domain.enums.HeatExchangerNetworkDesignMethod` enum is the
single method identity used for dispatch and result metadata.
``design.result.manifest.method_sequence`` records the executed task sequence.
For tiered OpenHENS runs, the manifest also records the synthesis quality tier,
selected pathway, and protected-fallback status.

``HENS_SYNTHESIS_QUALITY_TIER`` remains a persistent configuration field with a
default of tier 1 for prepared-problem workflows. User code should prefer
``problem.design.enhanced_heat_exchanger_network(quality_tier=...)`` for method-level
tier selection because it applies a call-local override without mutating the
loaded problem configuration. Runtime ``options`` passed to design accessors are
reserved for runtime context and do not accept persistent ``HENS_*`` overrides.

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
