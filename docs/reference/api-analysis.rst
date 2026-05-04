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

1. :mod:`OpenPinch.services.data_preparation` validates options, normalises the
   zone hierarchy, and constructs :class:`~OpenPinch.classes.zone.Zone`,
   :class:`~OpenPinch.classes.stream.Stream`, and
   :class:`~OpenPinch.classes.stream_collection.StreamCollection` objects.
2. :mod:`OpenPinch.services.common.problem_table_analysis` builds the shifted and
   real-temperature problem tables used throughout the rest of the workflow.
3. :mod:`OpenPinch.services.direct_heat_integration.direct_integration_entry` computes direct
   integration targets for unit-operation and process zones.
4. :mod:`OpenPinch.services.indirect_heat_integration.indirect_integration_entry` aggregates solved
   subzones into site-style indirect integration targets when the hierarchy
   requires it.
5. :mod:`OpenPinch.services.common.graph_data` converts solved tables and targets into
   serialisable graph payloads for reporting and Streamlit visualisation.

Preparation and Zone Construction
---------------------------------

These functions are the bridge between external schema payloads and the
internal object model.

.. automodule:: OpenPinch.services.data_preparation
   :members:

Direct and Indirect Targeting Entrypoints
-----------------------------------------

These modules own the top-level targeting workflows once a zone tree has been
constructed.

- Direct integration works on process streams within a zone and applies problem
  table analysis, utility targeting, optional heat-pump targeting, and optional
  cost/exergy add-ons.
- Indirect integration aggregates the net thermal behaviour of solved subzones
  and applies utility-to-utility balancing for total-site style studies.
- Lower-level heat-pump and refrigeration screening for both routes is
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
  other grand-composite-curve variants used for interpretation and advanced
  targeting.
- :mod:`OpenPinch.services.common.graph_data` translates tables and targets into the
  graph structures emitted in :class:`~OpenPinch.lib.schema.TargetOutput`.

.. automodule:: OpenPinch.services.common.problem_table_analysis
   :members:

.. automodule:: OpenPinch.services.common.utility_targeting
   :members:

.. automodule:: OpenPinch.services.common.gcc_manipulation
   :members:

.. automodule:: OpenPinch.services.common.graph_data
   :members:

Advanced Analyses
-----------------

The modules below expose specialised calculations that sit on top of the core
problem-table workflow. Some are used automatically when corresponding options
are enabled, while others are better viewed as expert-level helper libraries.

Heat Pump Targeting API
~~~~~~~~~~~~~~~~~~~~~~~

The main user-facing screening workflow is
:meth:`OpenPinch.classes.pinch_problem.PinchProblem.evaluate_heat_pump_integration`.
The module below exposes the lower-level targeting helpers used by the direct
and indirect integration entrypoints, plus advanced plotting helpers for solved
multi-cycle results.

Current public helpers include:

- :func:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry.get_direct_heat_pump_and_refrigeration_target`
- :func:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry.get_indirect_heat_pump_and_refrigeration_target`
- :func:`OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry.plot_multi_hp_profiles_from_results`

.. automodule:: OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry
   :members:

.. automodule:: OpenPinch.services.common.capital_cost_and_area_targeting
   :members:

.. automodule:: OpenPinch.services.common.temperature_driving_force
   :members:

.. automodule:: OpenPinch.services.power_cogeneration_analysis
   :members:

.. automodule:: OpenPinch.services.exergy_analysis.exergy_targeting_entry
   :members:

Legacy Research Module
~~~~~~~~~~~~~~~~~~~~~~

:mod:`OpenPinch.services.energy_transfer_analysis.energy_transfer_analysis` is retained in the source
tree as an experimental placeholder from older research code. It is not part of
the supported public workflow and is therefore intentionally omitted from the
generated API listing here.
