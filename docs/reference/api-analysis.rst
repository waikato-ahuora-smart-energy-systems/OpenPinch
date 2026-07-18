Analysis Package
================

This is an unsupported contributor reference. Analysis modules may move or
change without a compatibility facade; external callers should use
:class:`OpenPinch.PinchProblem`.

The analysis package contains the numerical workflow that turns validated
inputs into pinch targets, utility allocations, and graph-ready composite
curve data. The modules are designed so that the high-level service layer can
use them in sequence, but they can also be called directly for custom studies
or research workflows.

Pipeline Overview
-----------------

The analysis stack typically runs in this order:

1. :mod:`OpenPinch.application._problem.input.construction` validates
   options, normalises the zone hierarchy, and constructs
   :class:`~OpenPinch.domain.zone.Zone`,
   :class:`~OpenPinch.domain.stream.Stream`, and
   :class:`~OpenPinch.domain.stream_collection.StreamCollection` objects.
2. :mod:`OpenPinch.application.targeting` selects the orchestration path used
   by the high-level service layer and ``PinchProblem.target.*`` helpers.
3. :mod:`OpenPinch.analysis.targeting.cascade` builds the shifted
   and real temperature problem tables used throughout the rest of the
   workflow.
4. :mod:`OpenPinch.analysis.targeting.direct`
   computes direct integration targets for unit-operation and process zones.
5. :mod:`OpenPinch.analysis.targeting.total_site`
   aggregates solved subzones into site-style indirect integration targets when
   the hierarchy requires it.
6. :mod:`OpenPinch.analysis.graphs.service` converts solved tables and
   targets into serialisable graph data for reporting and Streamlit
   visualisation.

Service Package Map
-------------------

.. automodule:: OpenPinch.analysis
   :no-members:

.. automodule:: OpenPinch.application.targeting
   :members:

Preparation and Zone Construction
---------------------------------

These functions are the bridge between external schema inputs and the
internal object model.

.. automodule:: OpenPinch.application._problem.input.construction
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
  :mod:`OpenPinch.analysis.heat_pumps.service`.

.. automodule:: OpenPinch.analysis.targeting.direct
   :members:

.. automodule:: OpenPinch.analysis.targeting.total_site
   :members:

Problem Tables, Utility Allocation, and Graph Data
--------------------------------------------------

These modules implement the numerical building blocks that the entry-point
workflows depend on.

- :mod:`OpenPinch.analysis.targeting.cascade` generates the cascade tables
  and extracts pinch, utility, and heat-recovery targets from them.
- :mod:`OpenPinch.analysis.targeting.utilities` assigns multiple utilities across
  heating and cooling deficits while respecting temperature feasibility.
- :mod:`OpenPinch.analysis.targeting.grand_composite` derives pocket-free, assisted, and
  other Grand Composite Curve variants used for interpretation and advanced
  targeting.
- :mod:`OpenPinch.analysis.graphs.service` translates tables and targets into the
  graph structures emitted in :class:`~OpenPinch.contracts.output.TargetOutput`.

.. automodule:: OpenPinch.analysis.targeting.cascade
   :members:

.. automodule:: OpenPinch.analysis.targeting.utilities
   :members:

.. automodule:: OpenPinch.analysis.targeting.grand_composite
   :members:

.. automodule:: OpenPinch.analysis.graphs.service
   :members:

Process Component Services
--------------------------

Process components mutate a prepared in-memory model before a target is
rerun. They sit below ``PinchProblem.components`` and are most useful for
workspace before/after studies where the active stream set changes between
cases.

.. automodule:: OpenPinch.analysis.heat_pumps
   :no-members:
   :no-index:

.. automodule:: OpenPinch.analysis.heat_pumps.components
   :members:
   :no-index:

.. automodule:: OpenPinch.analysis.heat_pumps.process_mvr
   :members:
   :no-index:

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr
   :no-members:

.. automodule:: OpenPinch.analysis.heat_pumps.direct_mvr.execution
   :members: solve_direct_gas_mvr_stream, coerce_positive_mvr_stage_count
   :no-index:

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
:mod:`OpenPinch.analysis.heat_pumps.service`.

.. automodule:: OpenPinch.analysis.targeting.area_cost
   :members:

.. automodule:: OpenPinch.analysis.targeting.temperature_driving_force
   :members:

.. automodule:: OpenPinch.analysis.power.service
   :members:

Experimental or Partial Analysis Modules
----------------------------------------

The modules below remain visible for codebase orientation and restoration work,
but they should not be read as stable production workflows. They are present in
the repository with partial implementations, commented stubs, or incomplete
workflow documentation.

.. automodule:: OpenPinch.analysis.exergy.service
   :no-members:
