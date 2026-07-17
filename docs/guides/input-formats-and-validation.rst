Input Formats and Validation
============================

.. warning::

   Direct schema and validation-owner imports in this advanced guide are
   unsupported. Only :func:`OpenPinch.main.pinch_analysis_service` is
   compatibility protected.

Purpose
-------

Use this guide to choose the right input shape and understand where validation
fits before targeting begins.

Prerequisites
-------------

Start with :doc:`first-solve-python` if you have not run a packaged sample
case yet. For internal schema-first experiments, import models from
``OpenPinch.contracts.input`` and accept that their path may change.

Sample Case
-----------

Use ``basic_pinch.json`` for first validation checks and
``crude_preheat_train_multiperiod.json`` when you need named operating
periods.

Runnable Workflow
-----------------

Wrapper-based validation:

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json")
   validation = problem.validation_report()
   input_data = problem.validate()

Schema-first validation:

.. code-block:: python

   from OpenPinch.contracts.input import TargetInput

   source_data = {"streams": [...], "utilities": [...]}
   input_data = TargetInput.model_validate(source_data)

Expected Output
---------------

Validation returns typed input data or a structured validation report before
the runtime ``Zone`` hierarchy is prepared. Input data can be structurally
valid and still produce warnings about unusual thermal assumptions.

Supported Source Shapes
-----------------------

``PinchProblem`` and ``PinchWorkspace`` accept:

- packaged sample-case names such as ``basic_pinch.json``
- JSON files
- Excel workbooks such as ``.xlsx``, ``.xls``, ``.xlsb``, and ``.xlsm``
- CSV directories containing ``streams.csv`` and ``utilities.csv``
- ``(streams_csv, utilities_csv)`` tuples
- ``TargetInput`` instances
- plain mappings that already match the case-input structure

Canonical Input Fields
----------------------

Structured process-stream, segment, and temperature-profile mappings reject
unknown fields. Process streams use ``name`` and
``heat_capacity_flowrate`` as their canonical field names. Retired spellings
such as ``stream_name``, ``heat_capacity_flow_rate``, and
``flow_heat_capacity`` are invalid inputs and are not migrated.

Variable Heat-Capacity Streams
------------------------------

Structured Python and JSON inputs can describe one physical stream with an
ordered piecewise thermal profile. The prepared problem retains one parent
``Stream``; its internal child records are used for interval, area, and network
calculations.

Every ``Value`` exposed by a prepared parent or child segment is a read-only
view. Change domain state by assigning the stream property, calling
``set_value_attr_at_idx(...)``, or using ``update_segment(...)`` and
``update_segments(...)``. These APIs validate a mutable candidate and commit
the complete change transactionally.

Explicit segment input supplies each piece in physical traversal order. Every
segment target temperature must equal the next segment supply temperature.
OpenPinch preserves this order and rejects gaps, overlaps, reversals, and
non-positive segment duties or heat-transfer coefficients.

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem(
       {
           "streams": [
               {
                   "zone": "Site",
                   "name": "Variable CP feed",
                   "segments": [
                       {
                           "t_supply": 180.0,
                           "t_target": 140.0,
                           "heat_flow": 60.0,
                           "htc": 1.5,
                       },
                       {
                           "t_supply": 140.0,
                           "t_target": 80.0,
                           "heat_flow": 150.0,
                           "htc": 0.9,
                       },
                   ],
               }
           ],
           "utilities": [],
       },
       project_name="Site",
   )

A temperature--cumulative-heat profile is an alternative nested input. Its
points are authoritative: OpenPinch infers parent endpoints and duty, and
validates any duplicated parent values instead of rescaling the profile.
Cumulative heat must increase strictly. Temperature plateaus follow the
existing minimum sensible-temperature-span convention, while reversals are
rejected before linearisation.

.. code-block:: python

   stream_input = {
       "zone": "Site",
       "name": "Calculated profile",
       "profile": {
           "points": [
               {"temperature": 180.0, "cumulative_heat": 0.0},
               {"temperature": 140.0, "cumulative_heat": 60.0},
               {"temperature": 80.0, "cumulative_heat": 210.0},
           ],
           "linearisation_tolerance": 0.01,
       },
   }

Nested profiles are supported by Python objects, JSON, and workspace inputs.
Flat CSV and Excel stream rows remain unchanged and are never grouped by name
or adjacent temperatures.

Segmented Utilities and Prices
------------------------------

Structured utility inputs accept the same mutually exclusive ``segments`` or
``profile`` shapes. Explicit segments may each provide a different ``price``.
A segment price overrides the parent utility price; the parent price fills any
missing child price, and the existing utility default applies when neither is
provided. The prepared utility remains one parent stream, whose displayed
price is the duty-weighted effective value.

.. code-block:: python

   segmented_steam = {
       "name": "Segmented steam",
       "type": "Hot",
       "price": 40.0,
       "segments": [
           {
               "t_supply": 250.0,
               "t_target": 220.0,
               "heat_flow": 50.0,
               "price": 20.0,
           },
           {
               "t_supply": 220.0,
               "t_target": 180.0,
               "heat_flow": 100.0,
           },
       ],
   }

Here the first segment costs 20 per energy unit and the second inherits 40.
Temperature--heat ``profile`` input deliberately accepts one parent/default
price only; use explicit segments when interval prices differ.

Interpretation
--------------

Choose the source shape by ownership:

- Use packaged sample cases for learning and regression examples.
- Use JSON for version-controlled studies.
- Use workbooks when the source of truth is spreadsheet-oriented.
- Use CSV bundles when streams and utilities originate from separate tabular
  exports.
- Use schema-first Python inputs when another system constructs cases in
  memory.

Configuration belongs in ``TargetInput.options`` and is materialized as a
runtime ``Configuration`` object on prepared zones. Use
``config_options()`` to discover supported option keys.

Next Steps
----------

- :doc:`../api/schemas-and-config` for typed schema and option details.
- :doc:`../api/service-layer` for the preparation boundary.
- :doc:`zonal-and-total-site-workflows` when your input has a zone tree.
