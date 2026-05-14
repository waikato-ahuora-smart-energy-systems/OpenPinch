Heat Pump and Refrigeration Methods
===================================

OpenPinch treats heat-pump and refrigeration work as an integration question
first and a cycle question second.

Primary Question
----------------

The package is mainly asking:

- Does a candidate temperature lift improve the utility picture?

It is not primarily asking:

- Can a standalone cycle be solved in isolation?

This is why the user-facing workflows focus on utility targets, heat recovery,
and graph changes before diving into cycle-specific details.

Two Main Workflow Shapes
------------------------

Scenario comparison workflow
   Heat-pump studies are framed in the docs as before/after integration
   questions, but the current package surface exposes that work through the
   explicit HPR targeting methods rather than a dedicated wrapper helper.

Targeting workflow
   `problem.target.direct_heat_pump(...)`,
   `problem.target.indirect_heat_pump(...)`, and the refrigeration companion
   methods use the lower-level HPR targeting stack to screen or optimize
   integration opportunities more directly.

Direct vs Indirect HPR
----------------------

Direct HPR workflows act inside the targeted zone boundary.

Indirect HPR workflows act on a larger aggregated utility picture, which is
why they should be interpreted together with site-style graph views and utility
interactions.

Cycle Layer
-----------

The deeper HPR stack includes multiple thermodynamic cycle models and
optimization helpers. For many users this is implementation depth, not the
first layer to learn. The more important first interpretation is:

- which utilities are displaced
- which temperature lift is being bridged
- whether the graph picture improves in a meaningful way

Recommended Follow-On Pages
---------------------------

- :doc:`../guides/heat-pump-workflows`
- :doc:`../api/service-layer`
- :doc:`../api/generated-index`
