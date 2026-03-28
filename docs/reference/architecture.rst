Architecture and Methods
========================

This page summarises the design of OpenPinch in terms used throughout the
manuscript and the codebase.

Core Concepts
-------------

- ``Stream``: process or utility thermal stream with supply/target
  temperatures, heat flow, and transfer properties.
- ``Zone``: hierarchical system boundary for targeting (unit, process, site,
  community, region).
- ``ProblemTable``: temperature-interval cascade structure used to derive pinch
  temperatures, utility targets, and composite curves.
- ``EnergyTarget``: packaged targeting outcomes for a zone, including utility
  demands, recovery targets, and optional economic metrics.
- ``PinchProblem``: convenience orchestrator that loads input data, runs
  analysis, and exports results.

Multi-Scale Hierarchy
---------------------

OpenPinch follows nested thermodynamic scopes:

1. Unit operation
2. Process zone
3. Site
4. Community (planned extension)
5. Region (planned extension)

Each parent zone can aggregate subzone targets and run additional indirect
integration steps where appropriate.

Analysis Workflow
-----------------

1. Validate input payloads via :mod:`OpenPinch.lib.schema` models.
2. Build the zone tree and stream/utility collections with
   :func:`OpenPinch.analysis.data_preparation.prepare_problem`.
3. Run direct integration (problem-table and multi-utility targeting) with
   :func:`OpenPinch.analysis.direct_integration_entry.compute_direct_integration_targets`.
4. For site-style aggregation, run indirect integration with
   :func:`OpenPinch.analysis.indirect_integration_entry.compute_indirect_integration_targets`.
5. Package targets, utilities, and graph payloads for programmatic use and
   Excel export.

Direct and Indirect Integration
-------------------------------

Direct integration applies the classical problem-table cascade to process
streams and then assigns utilities against resulting deficits/surpluses.
Indirect integration reuses cascade logic at site level by operating on net
segments imported from subzones, then resolves utility-to-utility balancing.

Data Contracts
--------------

The schema layer provides strongly typed request/response payloads that enforce
consistent field names and units. This enables reproducible scripting and
clean integration with external tools and dashboards.
