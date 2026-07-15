Input Formats and Validation
============================

Purpose
-------

Use this guide to choose the right input shape and understand where validation
fits before targeting begins.

Prerequisites
-------------

Start with :doc:`first-solve-python` if you have not run a packaged sample
case yet. For schema-first usage, import the public models from
``OpenPinch.lib.schemas.io``.

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

   from OpenPinch.lib.schemas.io import TargetInput

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

Variable Heat-Capacity Streams
------------------------------

Structured Python and JSON inputs can describe one physical stream with an
ordered piecewise thermal profile. The prepared problem retains one parent
``Stream``; its children are ``StreamSegment`` objects used for interval,
area, and network calculations.

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
