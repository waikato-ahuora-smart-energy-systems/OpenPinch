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
   and active/base ``dt_cont`` behavior. A variable-heat-capacity parent owns
   an immutable ordered view of ``StreamSegment`` children while retaining one
   physical stream identity.

``StreamSegment``
   One ordered, locally linear thermal piece owned by a parent ``Stream``.
   Segment mutations are transactional and revalidate the complete profile.
   ``Stream.update_segments(...)`` applies sparse changes to several children
   in one atomic commit; an invalid index, attribute, or resulting profile
   leaves the parent and every child unchanged.

For segmented utilities, child prices may differ. The parent ``price`` is the
duty-weighted effective value for each operating period, so the derived parent
cost equals the sum of the child costs. Assigning ``parent.price`` is an
explicit broadcast to every child; updating one child afterwards may make the
prices differ again.

``StreamCollection``
   Ordered container with hot/cold filtering and utility inversion helpers.
   Ordinary iteration and reports remain parent-based; explicit expanded
   exports include canonical parent keys and ordered segment identities.

``ProblemTable``
   Numerical temperature-interval table behind composite curves, pinch
   temperatures, utility cascades, and several advanced targeting routines.

``ProcessComponent``
   Memory-only component attached to a prepared problem when the model needs
   to be mutated before targeting. The direct process MVR component uses this
   layer to replace selected hot gas/vapour streams with compressed
   replacement streams.

``HeatExchangerNetwork``
   Selected heat exchanger network design result with ordered exchanger
   records, total-duty helpers, and
   ``build_grid_diagram(...)`` for Plotly grid inspection.

``HeatExchanger``
   One physical parent-level match in a synthesized network. For segmented
   streams, ``segment_area_contributions`` contains ordered diagnostic slices;
   the exchanger exposes period duty and area totals plus the maximum
   period-total design area without treating those slices as topology nodes.

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

.. autoclass:: OpenPinch.classes.stream.StreamSegment
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.stream_collection.StreamCollection
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.problem_table.ProblemTable
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork
   :members:
   :no-index:

.. autoclass:: OpenPinch.classes.heat_exchanger.HeatExchanger
   :members:
   :no-index:

Process Components
------------------

Process components are attached to a live ``PinchProblem`` and are not part of
the external input schema. They are useful for before/after studies where a
specific unit operation changes the active process stream set before targeting.

.. autoclass:: OpenPinch.services.components.process_components.ProcessComponent
   :members:
   :no-index:

.. autoclass:: OpenPinch.services.components.process_mvr.ProcessMVRComponent
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
