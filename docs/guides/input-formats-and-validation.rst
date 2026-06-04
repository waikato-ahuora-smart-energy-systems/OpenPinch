Input Formats and Validation
============================

OpenPinch supports multiple input styles over the same analysis engine. This
guide explains which one to use and how validation fits into the workflow.

Supported Source Shapes
-----------------------

The high-level wrappers accept these source shapes today:

JSON file
   Best for reproducible version-controlled studies and programmatic
   generation.

Packaged sample-case name
   A ``*.json`` name such as ``basic_pinch.json`` or
   ``crude_preheat_train.json``. This resolves the packaged asset only when no
   local file with the same name exists.

Workbook file
   Best when starting from an established spreadsheet-oriented workflow. The
   loader accepts ``.xlsx``, ``.xls``, ``.xlsb``, and ``.xlsm``.

CSV directory
   A directory containing ``streams.csv`` and ``utilities.csv``.

CSV tuple
   A ``(streams_csv, utilities_csv)`` tuple when the two files are already
   known separately.

Schema-first Python input
   Best for application integration and typed construction in code.

Plain mapping
   Useful when another part of your application already assembled the payload
   in memory.

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

This means input data can be structurally valid but still produce warnings about
assumptions or unusual thermal conditions.

Recommended Validation Workflow
-------------------------------

Validation is a Python-side workflow. For wrapper-based usage:

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json")
   input_data = problem.validate()

For schema-first usage:

.. code-block:: python

   from OpenPinch.lib.schemas.io import TargetInput

   input_data = TargetInput.model_validate(payload)

Configuration Inputs
--------------------

Package configuration belongs on canonical option keys and the runtime
``zone.config`` fields built from those options.

Practical implications:

- use the documented ``TargetInput`` / ``Configuration`` field names
- use ``zone_tree`` when you need explicit hierarchy control
- use the high-level wrappers when you want validation context and source
  normalization handled for you

Preparation Boundary
--------------------

After validation, OpenPinch prepares a runtime
:class:`~OpenPinch.classes.zone.Zone` tree. That stage assigns streams to
zones, applies ``dt_cont_multiplier`` values from the zone hierarchy,
synthesizes utility placement context, and leaves the result ready for direct,
indirect, HPR, or cogeneration targeting.

If you need that boundary directly, use
:func:`OpenPinch.services.data_preprocessing_service`.

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

Use a packaged sample-case name when:

- you want a maintained known-good example without shipping example files in
  your own repository

Next Steps
----------

- For the user-facing service boundary, see :doc:`../api/service-layer`.
- For the main schemas and config model, see :doc:`../api/schemas-and-config`.
