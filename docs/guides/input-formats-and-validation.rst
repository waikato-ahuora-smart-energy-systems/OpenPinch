Input Formats and Validation
============================

OpenPinch supports multiple input styles over the same analysis engine. This
guide explains which one to use and how validation fits into the workflow.

Supported Input Shapes
----------------------

JSON payload
   Best for reproducible version-controlled studies and programmatic
   generation.

Workbook input
   Best when starting from an established spreadsheet-oriented workflow.

CSV bundle
   Best when stream and utility data come from separate tabular export paths.

Schema-first Python input
   Best for application integration and typed construction in code.

Core Data Contract
------------------

The main programmatic contract is:

- :class:`OpenPinch.lib.schemas.io.TargetInput`
- :class:`OpenPinch.lib.schemas.io.StreamSchema`
- :class:`OpenPinch.lib.schemas.io.UtilitySchema`
- :class:`OpenPinch.lib.schemas.io.ZoneTreeSchema`

At minimum, a solve requires process streams. Utilities and zone hierarchies
may be explicit or synthesized depending on the workflow.

Validation Stages
-----------------

OpenPinch validation happens in layers:

1. schema validation
2. semantic validation and warnings
3. input preparation into a runtime `Zone` tree

This means a payload can be structurally valid but still produce warnings about
assumptions or unusual thermal conditions.

Recommended Validation Workflow
-------------------------------

Validation is a Python-side workflow:

.. code-block:: python

   problem = PinchProblem("basic_pinch.json")
   payload = problem.validate()

Configuration Inputs
--------------------

Package configuration belongs on canonical option keys and runtime
`zone.config` fields. Legacy workbook-style side gateways are intentionally no
longer the preferred package contract.

Important consequence:

- use canonical option names
- treat unknown option names as real errors
- use `zone_tree` when you need explicit hierarchy control

When To Use Which Input Style
-----------------------------

Use JSON when:

- you want stable, explicit, reviewable inputs

Use workbook input when:

- the source of truth is still spreadsheet-driven

Use schema-first inputs when:

- you are integrating OpenPinch into another Python system

Use CSV bundles when:

- streams and utilities originate from separate tabular data exports

Next Steps
----------

- For the user-facing service boundary, see :doc:`../api/service-layer`.
- For the main schemas and config model, see :doc:`../api/schemas-and-config`.
