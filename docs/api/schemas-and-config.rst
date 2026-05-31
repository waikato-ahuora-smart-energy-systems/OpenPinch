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

Target Models
-------------

Solved targets are normalized through the target schema layer before they are
returned to users or exported.

.. autoclass:: OpenPinch.lib.schemas.targets.BaseTargetModel
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
