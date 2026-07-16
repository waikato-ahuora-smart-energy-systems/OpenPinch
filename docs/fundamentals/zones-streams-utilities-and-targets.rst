Zones, Streams, Utilities, and Targets
======================================

OpenPinch is built around a small set of domain concepts. Understanding these
objects makes the rest of the package easier to reason about.

Streams
-------

A `Stream` represents a thermal source or sink with:

- supply and target temperatures
- heat flow
- heat-transfer information such as `htc`
- temperature-approach assumptions such as `dt_cont`

Process streams are the primary objects used to build direct pinch targets.
Utility streams are used to model heating and cooling services explicitly when
the problem definition includes them.

Stream Collections
------------------

`StreamCollection` is the ordered container used throughout the targeting
workflow. Collections are used to:

- group hot or cold streams
- separate process and utility contexts
- support iteration, filtering, and graph generation

The collection itself is deliberately lightweight. Most of the thermodynamic
meaning still lives on the individual `Stream` objects and on the surrounding
zone context.

Zones
-----

A `Zone` is the hierarchy node that defines an analysis boundary.

In practice a zone may represent:

- a unit operation
- a process area
- an entire site

A zone owns:

- subzones
- stream collections
- utilities
- configuration state
- solved target models

This is what allows OpenPinch to move from a single process-zone problem to a
Total Site aggregation workflow.

Targets
-------

Solved targets are stored as validated models attached to zones. These targets
carry:

- utility targets
- recovery metrics
- pinch temperatures
- graph data and related metadata
- optional advanced fields for HPR or cogeneration workflows

For most users, these targets are surfaced through:

- `problem.summary_frame()`
- `problem.results`
- `problem.plot.*`

Hierarchy and Scope
-------------------

The main modeling discipline is scope.

- Streams belong to zones.
- Zones can aggregate subzones.
- Direct targets usually describe local recovery inside a zone.
- Indirect targets usually describe higher-level recovery across solved
  subzones.

When interpreting results, always check which zone and which target type a row
belongs to before comparing values.

Zone and Target Hierarchy
-------------------------

The main in-memory hierarchy can be sketched as:

.. code-block:: text

   Site Zone
   |- Area / Process Zone
   |  |- hot streams
   |  |- cold streams
   |  |- hot / cold utilities
   |  `- targets
   |     |- Direct Integration
   |     |- Indirect / Total Site views
   |     |- HPR / refrigeration targets
   |     `- cogeneration / area-cost post-processing
   `- Subzones
      `- repeat the same pattern recursively

This is why a result row is never just a number. It is always a number tied to
both a zone scope and a target family.

Useful Mental Model
-------------------

.. code-block:: text

   TargetInput
       -> StreamSchema / UtilitySchema / ZoneTreeSchema
       -> prepare_problem(...)
       -> Zone tree
       -> target models on zones
       -> TargetOutput / graphs / summaries / exports

Recommended Follow-On Pages
---------------------------

- :doc:`problem-table-and-temperature-shifting`
- :doc:`direct-vs-indirect-integration`
- :doc:`../api/domain-model`
