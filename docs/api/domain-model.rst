Domain Model
============

This page is an unsupported contributor reference. Runtime domain classes and
parent-owned records may change without compatibility aliases.

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
   an immutable ordered view of internal segment records while retaining one
   physical stream identity.

Segment mutations are transactional and revalidate the complete profile.
``Stream.update_segments(...)`` applies sparse changes to several children in
one atomic commit; an invalid index, attribute, or resulting profile leaves the
parent and every child unchanged. Runtime segment record classes are private;
construct them through ``Stream`` mappings or ``StreamSegmentSchema`` inputs.

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
   records and period-aware total-duty helpers. The presentation-owned
   ``build_grid_diagram(...)`` function handles Plotly grid inspection. Period
   identity may be omitted only when the network has exactly one period.

``HeatExchanger``
   One physical parent-level match in a synthesized network. For segmented
   streams, ``segment_area_contributions`` contains ordered diagnostic slices;
   shared topology, maximum design area, and capital data remain on the
   exchanger. Operating data is read from ``state(period_id)``.

Operating-period records are owned by each ``HeatExchanger`` and contain duty,
activity, terminal approaches, branch split fractions, and source/sink inlet
and outlet temperatures. Their runtime classes are private; multiperiod access
always names the period through ``exchanger.state(period_id)``.

These are the objects you inspect when you need to understand how a case was
prepared or why a target changed after mutating the in-memory model.

Key Classes
-----------

.. autoclass:: OpenPinch.domain.zone.Zone
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.stream.Stream
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.stream_collection.StreamCollection
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.problem_table.ProblemTable
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.heat_exchanger_network.HeatExchangerNetwork
   :members:
   :no-index:

.. autoclass:: OpenPinch.domain.heat_exchanger.HeatExchanger
   :members:
   :no-index:

Process Components
------------------

Process components are attached to a live ``PinchProblem`` and are not part of
the external input schema. They are useful for before/after studies where a
specific unit operation changes the active process stream set before targeting.

.. autoclass:: OpenPinch.analysis.heat_pumps.components.ProcessComponent
   :members:
   :no-index:

.. autoclass:: OpenPinch.analysis.heat_pumps.process_mvr.ProcessMVRComponent
   :members:
   :no-index:

Solved Target Records
---------------------

Targets are stored on zones and normalized through schema models before export.
The base target schema is a useful reference when you are programmatically
comparing cases or consuming target results in another tool.

.. autoclass:: OpenPinch.domain.targets.BaseTargetModel
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
