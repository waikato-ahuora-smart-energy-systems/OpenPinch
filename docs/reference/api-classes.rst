Domain Classes
==============

This is an unsupported contributor reference. Runtime domain classes,
including parent-owned records, are not external construction contracts.

The class layer is the in-memory model behind the OpenPinch workflow. Most of
these objects are created for you by
:func:`~OpenPinch.application._problem.input.construction.prepare_problem`, but they are also
useful directly when building tests, custom workflows, or post-processing
studies.

How These Objects Fit Together
------------------------------

- :class:`~OpenPinch.domain.stream.Stream` and
  :class:`~OpenPinch.domain.stream_collection.StreamCollection` represent the
  thermal streams and ordered stream sets used to build problem tables.
- :class:`~OpenPinch.domain.zone.Zone` groups streams, utilities, targets, and
  subzones into a hierarchical model of the process, site, or wider system.
- :class:`~OpenPinch.domain.problem_table.ProblemTable` stores the numerical
  temperature-interval cascade that drives pinch and utility calculations.
- :class:`~OpenPinch.domain.targets.BaseTargetModel` stores one solved set
  of metrics for a zone and is later serialised into the main-service output.
- :class:`~OpenPinch.domain.value.Value` wraps scalar and discrete-period
  quantities with units for report-friendly serialisation.

.. automodule:: OpenPinch.domain
   :no-members:
   :no-index:

Streams and Collections
-----------------------

These are the most commonly manipulated domain objects outside the top-level
service layer.

.. automodule:: OpenPinch.domain.stream
   :members:

.. automodule:: OpenPinch.domain.stream_collection
   :members:

Zones, Targets, and Tables
--------------------------

These classes represent the solved hierarchy and its numerical results.

.. automodule:: OpenPinch.domain.zone
   :members:

.. automodule:: OpenPinch.domain.problem_table
   :members:

Heat Exchanger Network Design Records
-------------------------------------

These classes are OpenPinch-native internal result models for heat exchanger
network design outcomes. They expose exchanger links by source and sink stream
identity; raw solver axis positions remain lower-level implementation details.

.. automodule:: OpenPinch.domain.heat_exchanger
   :members:

.. automodule:: OpenPinch.domain.heat_exchanger_network
   :members:

Heat Exchanger Network Unit Models
----------------------------------

The HEN synthesis unit-model modules sit below the internal design accessors.
They are useful when inspecting how the pinch-design and stagewise equations
are assembled, but users normally call the methods through
``problem.design``.

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models
   :no-members:
   :no-index:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.base
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.stagewise
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.packed_pinch_design
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.packed_stagewise
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.stage_packing
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.models.problem
   :members:

Units and Scalar Helpers
------------------------

:class:`~OpenPinch.domain.value.Value` supports both ordinary scalar
quantities and discrete-period values with ``period_ids`` and normalised
``weights``. This makes it suitable for both deterministic reports and
period-weighted scenario data.

.. automodule:: OpenPinch.domain.value
   :members:

Process Component Models
------------------------

Process components are live model mutations attached after preparation and
before rerunning targets. The direct process MVR component owns the original
stream records, replacement streams, per-period stage results, and
activation/deactivation state used by workspace comparison studies.

.. automodule:: OpenPinch.analysis.heat_pumps.components
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.process_mvr
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr.execution
   :members: solve_direct_gas_mvr_stream, coerce_positive_mvr_stage_count

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr.models
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr.thermodynamics
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr.units
   :members:

Thermal Cycle and Cogeneration Unit Models
------------------------------------------

These classes support the advanced Heat Pump, refrigeration, and utility system
workflows documented in
:mod:`OpenPinch.analysis.heat_pumps.service`. They are
primarily useful for advanced users who want to inspect or construct detailed
cycle configurations directly.

.. automodule:: OpenPinch.analysis.heat_pumps.cycles
   :no-members:
   :no-index:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.carnot_cycles
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.mechanical_vapour_recompression_cycle
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.cycles.brayton_heat_pump
   :members:

.. automodule:: OpenPinch.analysis.power.steam_turbine
   :members:
