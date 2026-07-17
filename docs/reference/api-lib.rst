Schema and Configuration
========================

These direct contract and domain imports are unsupported contributor APIs.
Their wire behaviour is protected only when reached through
:func:`OpenPinch.main.pinch_analysis_service`.

The :mod:`OpenPinch.contracts` and :mod:`OpenPinch.domain` packages define the
typed boundary between callers and the analysis engine. Contracts describe
accepted inputs and returned outputs; domain modules own runtime values,
enumerated labels, and configuration.

Configuration and Numeric Conventions
-------------------------------------

:mod:`OpenPinch.domain.configuration` provides the runtime configuration object and the
global constants used across multiple modules. Many option names intentionally
mirror the long-standing Excel workbook so workbook and Python workflows can
share the same mental model.

.. automodule:: OpenPinch.contracts
   :no-members:
   :no-index:

.. automodule:: OpenPinch.domain.configuration
   :members:

Enumerations and Labels
-----------------------

The enums module centralises the canonical names used for zones, targets,
streams, graph series, heat exchanger network design methods, and
workbook-compatible option keys. Refer to these when you need deterministic
identifiers instead of free-form strings. Heat exchanger network synthesis uses
``HeatExchangerNetworkDesignMethod`` for both internal dispatch and task/result
method metadata, with ``HENDesignMethod`` as the shorter alias.

.. automodule:: OpenPinch.domain.enums
   :members:

Pydantic Schemas
----------------

The schema layer is the primary programmatic interface to OpenPinch. It is now
split by concern under :mod:`OpenPinch.contracts` rather than concentrated in
one large module.

The main schema modules are:

- :mod:`OpenPinch.contracts.common` for shared primitives such as
  ``ValueWithUnit``
- :mod:`OpenPinch.contracts.graphs` for graph data contracts
- :mod:`OpenPinch.contracts.input` for request and response models
- :mod:`OpenPinch.contracts.reporting` for summary and report-facing models
- :mod:`OpenPinch.contracts.hpr` for lower-level heat pump solver data models
- :mod:`OpenPinch.contracts.synthesis` for heat exchanger network synthesis
  method input/output, task, manifest, optional export, and design-result
  result data
- :mod:`OpenPinch.domain.targets` for runtime target models stored on
  solved zones
- :mod:`OpenPinch.contracts.turbine` for turbine solve result models

Common request models include:

- :class:`~OpenPinch.contracts.input.TargetInput`
- :class:`~OpenPinch.contracts.input.StreamSchema`
- :class:`~OpenPinch.contracts.input.UtilitySchema`
- :class:`~OpenPinch.contracts.input.ZoneTreeSchema`

Common response and reporting models include:

- :class:`~OpenPinch.contracts.output.TargetOutput`
- :class:`~OpenPinch.contracts.reporting.TargetResults`
- :class:`~OpenPinch.contracts.graphs.GraphSet`
- :class:`~OpenPinch.contracts.graphs.Graph`
- :class:`~OpenPinch.contracts.graphs.Segment`
- :class:`~OpenPinch.contracts.graphs.DataPoint`

Heat pump integration helper models include:

- :class:`~OpenPinch.contracts.hpr.HeatPumpTargetOutputs`

Specialised helper models also capture lower-level Heat Pump optimisation
inputs/outputs, integrated Heat Pump screening comparisons, piecewise stream
linearisation requests, and structured targeting/graph outputs.

.. automodule:: OpenPinch.contracts.common
   :members:

.. automodule:: OpenPinch.contracts.graphs
   :members:

.. automodule:: OpenPinch.contracts.hpr
   :members:

.. automodule:: OpenPinch.contracts.input
   :members:

.. automodule:: OpenPinch.contracts.reporting
   :members:

.. automodule:: OpenPinch.contracts.synthesis
   :members:

.. automodule:: OpenPinch.domain.targets
   :members:

.. automodule:: OpenPinch.contracts.turbine
   :members:
