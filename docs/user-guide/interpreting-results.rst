Interpreting Results
====================

This guide focuses on the question users usually have after a successful run:
what do the numbers and graphs mean, and what should be compared first?

Summary Metrics
---------------

The compact summary from :meth:`OpenPinch.classes.pinch_problem.PinchProblem.summary_frame`
is the fastest way to assess a case.

``Hot Utility Target``
   Minimum external heating demand. A lower value means the process needs less
   purchased or generated heating.

``Cold Utility Target``
   Minimum external cooling demand. A lower value means less heat must be
   rejected to cooling water, air, refrigeration, or another cold utility.

``Heat Recovery``
   Heat recovered internally within the targeted system. A higher value usually
   indicates a better-integrated case, but it should always be read together
   with the utility targets.

``Hot Pinch`` and ``Cold Pinch``
   Temperatures that identify the constrained region of the problem. They are
   the first place to look when the utility targets seem hard to improve.

``Degree of Integration``
   Ratio between the achieved recovery and the thermodynamic recovery limit. It
   is most useful when comparing two scenarios built on the same problem.

Recommended reading order:

1. Look at the main process or plant row first.
2. Compare hot and cold utility targets.
3. Check whether heat recovery moved in the expected direction.
4. Use the pinch temperatures to understand where the constraint remains.

Graph Interpretation
--------------------

Composite curves
   Show the overall heat-source and heat-sink profiles. Use them to judge the
   broad overlap between available hot duty and required cold duty.

Shifted composite curves
   Apply the minimum approach temperature. Use them to see the practical
   recovery picture rather than the purely unshifted overlap.

Balanced composite curves
   Support more detailed exchanger-network and area-oriented interpretation.
   They are useful once the main utility picture is already understood.

Grand composite curves
   Show the residual heat load after direct recovery. This is usually the first
   graph to inspect for utility selection, utility-level placement, and
   heat-pump opportunity identification.

Total-site profiles
   Aggregate multiple subzones. Use them to see whether site-level integration
   shifts utility demand between process areas.

Site utility grand composite curves
   Focus on utility-system interaction at the total-site level. They are the
   right view when comparing indirect integration scenarios.

Common reading mistakes:

- comparing site-level graphs to process-level summaries without checking the target row names
- treating a graph improvement as sufficient without confirming the utility targets also improved
- focusing on the detailed graph shape before checking the main hot and cold utility numbers

Heat-Pump Targeting And Integration
-----------------------------------

For OpenPinch users, heat-pump work should be interpreted as an integration
question first and a cycle question second.

The main questions are:

- which temperature lift is being bridged
- which external utilities are being displaced
- whether both hot and cold utility targets move in a useful direction
- how the grand composite curve changes after introducing the integration scenario

Cycle-level values such as condenser duty, evaporator duty, or compressor power
matter, but they are supporting context. The first decision is whether the
integration target improves the overall thermal picture.

The packaged ``04_heat_pump_workflow.ipynb`` notebook is organized around this
comparison:

1. solve the base case
2. introduce a candidate heat-pump integration scenario
3. compare utility targets and heat recovery
4. compare the before-and-after grand composite curves

For the concrete helper API and packaged sample, see :doc:`heat-pump-targeting`.

Comparing Zones And Cases
-------------------------

When comparing zones or scenarios:

1. compare like for like: process rows with process rows, site rows with site rows
2. compare hot and cold utility targets before comparing graph details
3. confirm that any increase in heat recovery corresponds to a meaningful utility benefit
4. use graphs to explain why the numbers changed, not as a substitute for the numbers

Output Surfaces
---------------

Terminal summary
   Best for quick verification and case-to-case comparison.

Excel export
   Best for detailed review, reporting, and utility breakdown inspection.

Graph HTML export
   Best for sharing visual outputs and reviewing curve shapes outside Python.

Notebook workflows
   Best for learning how to interpret outputs in context, especially for graph
   reading, zonal comparison, and heat-pump integration.
