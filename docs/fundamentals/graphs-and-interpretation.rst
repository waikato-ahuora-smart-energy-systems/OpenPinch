Graphs and Interpretation
=========================

OpenPinch is not only a target calculator. It is also a graph-oriented
interpretation toolkit. The graph families each answer a different question.

Reading Order
-------------

A good default sequence is:

1. read the summary metrics first
2. inspect the grand composite curve when utility choices matter
3. inspect composite or shifted composite curves when overlap and pinch
   behavior matter
4. inspect site-level profiles only after confirming you are comparing the
   correct target scope

Composite Curves
----------------

Composite curves show the aggregate hot and cold thermal profiles. They help
answer:

- How much broad overlap exists between heat sources and sinks?
- Where do the source and sink envelopes separate strongly?

Shifted Composite Curves
------------------------

Shifted composite curves apply the temperature-approach assumption used in the
targeting calculations. They are usually more decision-relevant than the raw
composite view because they reflect the practical recovery assumption.

Balanced Composite Curves
-------------------------

Balanced composite curves are useful when you want a more detailed exchanger
network or area-oriented interpretation after the primary utility picture is
already understood.

Grand Composite Curves
----------------------

The grand composite curve is usually the first graph to inspect when you are
thinking about:

- utility selection
- utility level placement
- Heat Pump opportunity screening
- residual heating or cooling pockets

It is often the best visual companion to the hot and cold utility targets.

Total Site Profiles and SUGCC
-----------------------------

At larger system scope, OpenPinch also supports:

- Total Site profiles
- site utility grand composite curves

These views are especially relevant when direct and indirect integration
answers differ and you need to understand utility system interaction across
subzones.

Common Interpretation Errors
----------------------------

- comparing a process-level summary row to a site-level graph
- focusing on graph shape before checking utility targets
- treating a graph improvement as sufficient without confirming the metrics
  improved
- forgetting that shifted graphs depend on the active temperature-approach
  assumptions

Where These Surfaces Appear
---------------------------

The main graph surfaces are:

- `problem.plot.composite_curve()`
- `problem.plot.shifted_composite_curve()`
- `problem.plot.balanced_composite_curve()`
- `problem.plot.grand_composite_curve()`
- `problem.plot.grand_composite_curve_with_heat_pump()`
- `problem.plot.net_load_profiles()`
- `problem.plot.export(...)`

After direct HPR targeting, the target-specific net load profile graph includes
the Heat Pump condenser and evaporator overlays, and the dedicated GCC with
Heat Pump view exposes the HPR cascade directly.

Recommended Follow-On Pages
---------------------------

- :doc:`../guides/graphing-and-interpretation`
- :doc:`../guides/heat-pump-workflows`
- :doc:`../guides/zonal-and-total-site-workflows`
