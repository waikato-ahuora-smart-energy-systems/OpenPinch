Schema and Configuration
========================

The :mod:`OpenPinch.lib` package defines the typed contract between external
callers and the analysis engine. If you are building an application around
OpenPinch, this is the package that tells you what inputs are accepted, what
outputs are returned, and which enumerated labels and configuration flags are
expected throughout the workflow.

Configuration and Numeric Conventions
-------------------------------------

:mod:`OpenPinch.lib.config` provides the runtime configuration object and the
global constants used across multiple modules. Many option names intentionally
mirror the long-standing Excel workbook so workbook and Python workflows can
share the same mental model.

.. automodule:: OpenPinch.lib.config
   :members:

Enumerations and Labels
-----------------------

The enums module centralises the canonical names used for zones, targets,
streams, graph series, and workbook-compatible option keys. Refer to these when
you need stable identifiers instead of free-form strings.

.. automodule:: OpenPinch.lib.enums
   :members:

Pydantic Schemas
----------------

The schema layer is the primary programmatic interface to OpenPinch. It is now
split by concern under :mod:`OpenPinch.lib.schemas` rather than concentrated in
one large module.

The main schema modules are:

- :mod:`OpenPinch.lib.schemas.common` for shared primitives such as
  ``ValueWithUnit``
- :mod:`OpenPinch.lib.schemas.graphs` for graph payload contracts
- :mod:`OpenPinch.lib.schemas.io` for public request and response payloads
- :mod:`OpenPinch.lib.schemas.reporting` for summary and report-facing models
- :mod:`OpenPinch.lib.schemas.hpr` for lower-level heat pump solver payloads
- :mod:`OpenPinch.lib.schemas.targets` for runtime target models stored on
  solved zones
- :mod:`OpenPinch.lib.schemas.turbine` for turbine solve result models

Common request models include:

- :class:`~OpenPinch.lib.schemas.io.TargetInput`
- :class:`~OpenPinch.lib.schemas.io.StreamSchema`
- :class:`~OpenPinch.lib.schemas.io.UtilitySchema`
- :class:`~OpenPinch.lib.schemas.io.ZoneTreeSchema`

Common response and reporting models include:

- :class:`~OpenPinch.lib.schemas.io.TargetOutput`
- :class:`~OpenPinch.lib.schemas.reporting.TargetResults`
- :class:`~OpenPinch.lib.schemas.graphs.GraphSet`
- :class:`~OpenPinch.lib.schemas.graphs.Graph`
- :class:`~OpenPinch.lib.schemas.graphs.Segment`
- :class:`~OpenPinch.lib.schemas.graphs.DataPoint`

Heat pump integration helper models include:

- :class:`~OpenPinch.lib.schemas.hpr.HeatPumpTargetOutputs`

Specialised helper models also capture lower-level heat pump optimisation
inputs/outputs, piecewise stream linearisation requests, and legacy
visualisation payloads.

.. automodule:: OpenPinch.lib.schemas
   :members:
