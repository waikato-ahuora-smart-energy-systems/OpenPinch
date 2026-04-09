Domain Classes
==============

The class layer is the in-memory model behind the OpenPinch workflow. Most of
these objects are created for you by
:func:`~OpenPinch.analysis.data_preparation.prepare_problem`, but they are also
useful directly when building tests, custom workflows, or post-processing
studies.

How These Objects Fit Together
------------------------------

- :class:`~OpenPinch.classes.stream.Stream` and
  :class:`~OpenPinch.classes.stream_collection.StreamCollection` represent the
  thermal streams and ordered stream sets used to build problem tables.
- :class:`~OpenPinch.classes.zone.Zone` groups streams, utilities, targets, and
  subzones into a hierarchical model of the process, site, or wider system.
- :class:`~OpenPinch.classes.problem_table.ProblemTable` stores the numerical
  temperature-interval cascade that drives pinch and utility calculations.
- :class:`~OpenPinch.classes.energy_target.EnergyTarget` stores one solved set
  of metrics for a zone and is later serialised into the public output schema.
- :class:`~OpenPinch.classes.value.Value` wraps quantities with units for
  report-friendly serialisation.

Streams and Collections
-----------------------

These are the most commonly manipulated domain objects outside the top-level
service layer.

.. automodule:: OpenPinch.classes.stream
   :members:

.. automodule:: OpenPinch.classes.stream_collection
   :members:

Zones, Targets, and Tables
--------------------------

These classes represent the solved hierarchy and its numerical results.

.. automodule:: OpenPinch.classes.zone
   :members:

.. automodule:: OpenPinch.classes.problem_table
   :members:

.. automodule:: OpenPinch.classes.energy_target
   :members:

Units and Scalar Helpers
------------------------

.. automodule:: OpenPinch.classes.value
   :members:

Heat Pump Classes
-----------------

These classes support the heat-pump targeting and cascade-construction
workflows documented in :mod:`OpenPinch.analysis.heat_pump_and_refrigeration_targeting`. They are
primarily useful for advanced users who want to inspect or construct detailed
cycle configurations directly.

.. automodule:: OpenPinch.classes.simple_heat_pump
   :members:

.. automodule:: OpenPinch.classes.multi_simple_heat_pump
   :members:

.. automodule:: OpenPinch.classes.cascade_heat_pump
   :members:

.. automodule:: OpenPinch.classes.brayton_heat_pump
   :members:
