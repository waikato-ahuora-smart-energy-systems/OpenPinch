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

1. :mod:`OpenPinch.analysis.data_preparation` validates options, normalises the
   zone hierarchy, and constructs :class:`~OpenPinch.classes.zone.Zone`,
   :class:`~OpenPinch.classes.stream.Stream`, and
   :class:`~OpenPinch.classes.stream_collection.StreamCollection` objects.
2. :mod:`OpenPinch.analysis.problem_table_analysis` builds the shifted and
   real-temperature problem tables used throughout the rest of the workflow.
3. :mod:`OpenPinch.analysis.direct_integration_entry` computes direct
   integration targets for unit-operation and process zones.
4. :mod:`OpenPinch.analysis.indirect_integration_entry` aggregates solved
   subzones into site-style indirect integration targets when the hierarchy
   requires it.
5. :mod:`OpenPinch.analysis.graph_data` converts solved tables and targets into
   serialisable graph payloads for reporting and Streamlit visualisation.

Preparation and Zone Construction
---------------------------------

These functions are the bridge between external schema payloads and the
internal object model.

.. automodule:: OpenPinch.analysis.data_preparation
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

.. automodule:: OpenPinch.analysis.direct_integration_entry
   :members:

.. automodule:: OpenPinch.analysis.indirect_integration_entry
   :members:

Problem Tables, Utility Allocation, and Graph Data
--------------------------------------------------

These modules implement the numerical building blocks that the entry-point
workflows depend on.

- :mod:`OpenPinch.analysis.problem_table_analysis` generates the cascade tables
  and extracts pinch, utility, and heat-recovery targets from them.
- :mod:`OpenPinch.analysis.utility_targeting` assigns multiple utilities across
  heating and cooling deficits while respecting temperature feasibility.
- :mod:`OpenPinch.analysis.gcc_manipulation` derives pocket-free, assisted, and
  other grand-composite-curve variants used for interpretation and advanced
  targeting.
- :mod:`OpenPinch.analysis.graph_data` translates tables and targets into the
  graph structures emitted in :class:`~OpenPinch.lib.schema.TargetOutput`.

.. automodule:: OpenPinch.analysis.problem_table_analysis
   :members:

.. automodule:: OpenPinch.analysis.utility_targeting
   :members:

.. automodule:: OpenPinch.analysis.gcc_manipulation
   :members:

.. automodule:: OpenPinch.analysis.graph_data
   :members:

Advanced Analyses
-----------------

The modules below expose specialised calculations that sit on top of the core
problem-table workflow. Some are used automatically when corresponding options
are enabled, while others are better viewed as expert-level helper libraries.

Heat Pump Targeting API
~~~~~~~~~~~~~~~~~~~~~~~

Public entry points exposed by ``OpenPinch.analysis.heat_pump_targeting``:

- :func:`OpenPinch.analysis.heat_pump_targeting.get_heat_pump_targets`
- :func:`OpenPinch.analysis.heat_pump_targeting.calc_heat_pump_cascade`
- :func:`OpenPinch.analysis.heat_pump_targeting.plot_multi_hp_profiles_from_results`

.. automodule:: OpenPinch.analysis.heat_pump_targeting
   :members:

.. automodule:: OpenPinch.analysis.capital_cost_and_area_targeting
   :members:

.. automodule:: OpenPinch.analysis.temperature_driving_force
   :members:

.. automodule:: OpenPinch.analysis.power_cogeneration_analysis
   :members:

.. automodule:: OpenPinch.analysis.exergy_targeting
   :members:

Legacy Research Module
~~~~~~~~~~~~~~~~~~~~~~

:mod:`OpenPinch.analysis.energy_transfer_analysis` is retained in the source
tree as an experimental placeholder from older research code. It is not part of
the supported public workflow and is therefore intentionally omitted from the
generated API listing here.
