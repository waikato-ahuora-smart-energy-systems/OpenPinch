Domain Classes
==============

The class layer is the in-memory model behind the OpenPinch workflow. Most of
these objects are created for you by
:func:`~OpenPinch.services.input_data_processing.data_preparation.prepare_problem`, but they are also
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
- :class:`~OpenPinch.lib.schemas.targets.BaseTargetModel` stores one solved set
  of metrics for a zone and is later serialised into the public output schema.
- :class:`~OpenPinch.classes.value.Value` wraps scalar and discrete-state
  quantities with units for report-friendly serialisation.

.. automodule:: OpenPinch.classes
   :no-members:

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

Units and Scalar Helpers
------------------------

:class:`~OpenPinch.classes.value.Value` supports both ordinary scalar
quantities and discrete-state values with ``state_ids`` and normalised
``weights``. This makes it suitable for both deterministic reports and
state-weighted scenario data.

.. automodule:: OpenPinch.classes.value
   :members:

Thermal Cycle and Cogeneration Unit Models
------------------------------------------

These classes support the advanced Heat Pump, refrigeration, and utility system
workflows documented in
:mod:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry`. They are
primarily useful for advanced users who want to inspect or construct detailed
cycle configurations directly.

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models
   :no-members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_cycle
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.parallel_vapour_compression_cycles
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.cascade_vapour_compression_cycle
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.carnot_cycles
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.mechanical_vapour_recompression_cycle
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_mvr_cascade
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.unit_models.brayton_heat_pump
   :members:

.. automodule:: OpenPinch.services.power_cogeneration.unit_models.multi_stage_steam_turbine
   :members:
