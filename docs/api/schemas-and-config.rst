Schemas and Config
==================

OpenPinch has two distinct but closely related typed surfaces:

- schema models for external inputs and returned results
- a runtime :class:`~OpenPinch.lib.config.Configuration` object attached to
  each prepared zone

Together they define the stable wire format and the per-zone analysis behavior
that the rest of the package consumes.

What Each Layer Does
--------------------

``TargetInput`` and related schemas
   Define the public request format for process streams, utilities, and the
   optional zone tree.

``TargetOutput`` and target/result schemas
   Define the structured response returned by the top-level service boundary.

``Configuration``
   Stores runtime knobs for targeting flags, heat pump parameters, utility
   assumptions, costing inputs, and turbine settings. Each prepared
   :class:`~OpenPinch.classes.zone.Zone` owns one config object.

Configuration
-------------

.. autoclass:: OpenPinch.lib.config.Configuration
   :members:
   :no-index:

Input and Output Schemas
------------------------

.. autoclass:: OpenPinch.lib.schemas.io.TargetInput
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.io.TargetOutput
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.io.StreamSchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.io.UtilitySchema
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.io.ZoneTreeSchema
   :members:
   :no-index:

Heat Exchanger Network Design Results
-------------------------------------

``TargetOutput.design`` stores a
:class:`~OpenPinch.lib.schemas.synthesis.HeatExchangerNetworkSynthesisResult`
after ``problem.design.enhanced_synthesis_method(quality_tier=...)``,
``problem.design.open_hens_method()``,
``problem.design.heat_exchanger_network_synthesis()``, or one of the direct
method accessors runs. The selected network is available as ``design.network``
and the ranked unique network candidates are available as
``design.ranked_networks``.

The same
:class:`~OpenPinch.lib.enums.HeatExchangerNetworkDesignMethod` enum is used for
public dispatch and task/result method metadata. ``HENDesignMethod`` is the
short public alias. ``design.design_method`` records the user-facing design
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
reserved for runtime state and do not accept persistent ``HENS_*`` overrides.

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
``design.network.build_grid_diagram(...)``. The standalone
:func:`OpenPinch.services.network_grid_diagram.build_grid_diagram` service still
accepts one or more
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork`
objects, and
``design.grid_diagram(solution_rank=...)`` remains available as a convenience
wrapper that selects a ranked network first. The returned object wraps the
Plotly ``fig``, a lightweight drawing adapter ``ax``, the selected ``network``,
and the normalized ``grid_model`` used to draw the topology.

.. autoclass:: OpenPinch.lib.schemas.synthesis.HeatExchangerNetworkSynthesisResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.synthesis.HeatExchangerNetworkSynthesisMethodInput
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.synthesis.HeatExchangerNetworkSynthesisMethodOutput
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.enums.HeatExchangerNetworkDesignMethod
   :members:
   :no-index:

Target Models
-------------

Solved targets are normalized through the target schema layer before they are
returned to users or exported.

.. autoclass:: OpenPinch.lib.schemas.targets.BaseTargetModel
   :members:
   :no-index:

HPR Schemas
-----------

The HPR schema layer carries the prepared configuration values, parsed backend
state, and simulated-cycle annualized cost accounting used by the targeting
services. Report-facing HPR cost fields use ``Value`` instances with public
units ``$`` and ``$/y``.

.. autoclass:: OpenPinch.lib.schemas.hpr.HeatPumpTargetInputs
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.hpr.HPRBackendResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.hpr.SimulatedHPRAnnualizedCostAccounting
   :members:
   :no-index:

Enums and Typed Constants
-------------------------

The :mod:`OpenPinch.lib` package also re-exports enums used across the public
API, including stream types, target labels, HPR cycle selectors, and turbine
model choices.

.. automodule:: OpenPinch.lib
   :no-members:
   :no-index:

Design Notes
------------

The schema layer should be the source of truth for external input contracts.
The configuration layer should be the source of truth for runtime toggles and
per-zone behavior. Keeping those roles distinct is what makes the package
predictable when used from notebooks, services, and the CLI.
