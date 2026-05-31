Domain Model
============

Once a problem is prepared, OpenPinch operates on a small set of in-memory
domain objects. Understanding these classes is the key to using the package as
more than a black box.

Core Objects
------------

``Zone``
   Hierarchical analysis boundary containing streams, utilities, targets, and
   graphs.

``Stream``
   Process or utility stream with supply/target states, shifted temperatures,
   and active/base ``dt_cont`` behavior.

``StreamCollection``
   Ordered container with hot/cold filtering and utility inversion helpers.

``ProblemTable``
   Numerical temperature-interval table behind composite curves, pinch
   temperatures, utility cascades, and several advanced targeting routines.

These are the objects you inspect when you need to understand how a case was
prepared or why a target changed after mutating the in-memory model.

Key Classes
-----------

.. autoclass:: OpenPinch.classes.zone.Zone
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.stream.Stream
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.stream_collection.StreamCollection
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.problem_table.ProblemTable
   :members:
   :no-index:

Solved Target Records
---------------------

Targets are stored on zones and normalized through schema models before export.
The base target schema is a useful reference when you are programmatically
comparing cases or consuming target results in another tool.

.. autoclass:: OpenPinch.lib.schemas.targets.BaseTargetModel
   :members:
   :no-index:

How These Objects Relate
------------------------

The usual flow is:

1. input schemas describe the external inputs
2. preparation turns those inputs into ``Zone`` and ``Stream`` objects
3. targeting populates ``ProblemTable`` objects, zone targets, and graph data
4. result schemas serialize the solved state back out

That layering is what lets the package support both high-level scripted use and
deeper programmatic inspection.
