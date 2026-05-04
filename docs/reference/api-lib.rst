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

The schema layer is the primary programmatic interface to OpenPinch.

Common request models include:

- :class:`~OpenPinch.lib.schema.TargetInput`
- :class:`~OpenPinch.lib.schema.StreamSchema`
- :class:`~OpenPinch.lib.schema.UtilitySchema`
- :class:`~OpenPinch.lib.schema.ZoneTreeSchema`

Common response and reporting models include:

- :class:`~OpenPinch.lib.schema.TargetOutput`
- :class:`~OpenPinch.lib.schema.TargetResults`
- :class:`~OpenPinch.lib.schema.GraphSet`
- :class:`~OpenPinch.lib.schema.Graph`
- :class:`~OpenPinch.lib.schema.Segment`
- :class:`~OpenPinch.lib.schema.DataPoint`

Heat-pump integration helper models include:

- :class:`~OpenPinch.lib.schema.HeatPumpIntegrationScenario`
- :class:`~OpenPinch.lib.schema.HeatPumpIntegrationComparison`

Specialised helper models also capture lower-level heat-pump optimisation
inputs/outputs, piecewise stream linearisation requests, and legacy
visualisation payloads.

.. automodule:: OpenPinch.lib.schema
   :members:
