Analysis Package
================

The analysis package contains the numerical workflow that turns validated
inputs into pinch targets, utility allocations, and graph-ready composite
curve data. The modules are designed so that the high-level service layer can
use them in sequence, but they can also be called directly for custom studies
or research workflows.

Pipeline Overview
-----------------

The analysis stack typically runs in this order:

1. :mod:`OpenPinch.services.input_data_processing.data_preparation` validates
   options, normalises the zone hierarchy, and constructs
   :class:`~OpenPinch.classes.zone.Zone`,
   :class:`~OpenPinch.classes.stream.Stream`, and
   :class:`~OpenPinch.classes.stream_collection.StreamCollection` objects.
2. :mod:`OpenPinch.services.services_entry` selects the orchestration path used
   by the high-level service layer and ``PinchProblem.target.*`` helpers.
3. :mod:`OpenPinch.services.common.problem_table_analysis` builds the shifted
   and real temperature problem tables used throughout the rest of the
   workflow.
4. :mod:`OpenPinch.services.direct_heat_integration.direct_integration_entry`
   computes direct integration targets for unit-operation and process zones.
5. :mod:`OpenPinch.services.indirect_heat_integration.indirect_integration_entry`
   aggregates solved subzones into site-style indirect integration targets when
   the hierarchy requires it.
6. :mod:`OpenPinch.services.common.graph_data` converts solved tables and
   targets into serialisable graph payloads for reporting and Streamlit
   visualisation.

Service Package Map
-------------------

.. automodule:: OpenPinch.services
   :no-members:

.. automodule:: OpenPinch.services.services_entry
   :members:

Preparation and Zone Construction
---------------------------------

These functions are the bridge between external schema inputs and the
internal object model.

.. automodule:: OpenPinch.services.input_data_processing
   :no-members:

.. automodule:: OpenPinch.services.input_data_processing.data_preparation
   :members:

Direct and Indirect Targeting Entrypoints
-----------------------------------------

These modules own the top-level targeting workflows once a zone tree has been
constructed.

- Direct integration works on process streams within a zone and applies Problem
  Table analysis, utility targeting, optional Heat Pump targeting, and optional
  cost/exergy add-ons.
- Indirect integration aggregates the net thermal behaviour of solved subzones
  and applies utility-to-utility balancing for Total Site studies.
- Lower-level Heat Pump and refrigeration screening for both routes is
  centralised in
  :mod:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry`.

.. automodule:: OpenPinch.services.direct_heat_integration.direct_integration_entry
   :members:

.. automodule:: OpenPinch.services.indirect_heat_integration.indirect_integration_entry
   :members:

Problem Tables, Utility Allocation, and Graph Data
--------------------------------------------------

These modules implement the numerical building blocks that the entry-point
workflows depend on.

- :mod:`OpenPinch.services.common.problem_table_analysis` generates the cascade tables
  and extracts pinch, utility, and heat-recovery targets from them.
- :mod:`OpenPinch.services.common.utility_targeting` assigns multiple utilities across
  heating and cooling deficits while respecting temperature feasibility.
- :mod:`OpenPinch.services.common.gcc_manipulation` derives pocket-free, assisted, and
  other Grand Composite Curve variants used for interpretation and advanced
  targeting.
- :mod:`OpenPinch.services.common.graph_data` translates tables and targets into the
  graph structures emitted in :class:`~OpenPinch.lib.schemas.io.TargetOutput`.

.. automodule:: OpenPinch.services.common.problem_table_analysis
   :members:

.. automodule:: OpenPinch.services.common.utility_targeting
   :members:

.. automodule:: OpenPinch.services.common.gcc_manipulation
   :members:

.. automodule:: OpenPinch.services.common.graph_data
   :members:

Advanced Add-On Analyses
------------------------

The modules below expose specialised calculations that sit on top of the core
Problem Table workflow. Some are used automatically when corresponding options
are enabled, while others are better viewed as expert-level helper libraries.

The Heat Pump and refrigeration stack is documented separately in
:doc:`api-heat-pump` because it spans a dedicated package with multiple cycle
optimisers and helper modules. The main low-level entrypoints remain
``compute_direct_heat_pump_or_refrigeration_target(...)`` and
``compute_indirect_heat_pump_or_refrigeration_target(...)`` in
:mod:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry`.

.. automodule:: OpenPinch.services.common.capital_cost_and_area_targeting
   :members:

.. automodule:: OpenPinch.services.common.temperature_driving_force
   :members:

.. automodule:: OpenPinch.services.power_cogeneration_analysis
   :members:

Experimental or Partial Analysis Modules
----------------------------------------

The modules below remain visible for codebase orientation and restoration work,
but they should not be read as stable production workflows. They are present in
the repository with partial implementations, commented stubs, or incomplete
workflow documentation.

.. automodule:: OpenPinch.services.exergy_analysis
   :no-members:

.. automodule:: OpenPinch.services.exergy_analysis.exergy_targeting_entry
   :no-members:
