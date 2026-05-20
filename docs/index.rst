OpenPinch Documentation
=======================

OpenPinch is a Python toolkit for advanced pinch analysis, direct and indirect
heat integration targeting, Total Site utility studies, graph generation,
Heat Pump integration screening, and cogeneration post-processing.

This documentation is organized to answer three different kinds of questions:

- What thermodynamic and workflow model does the package implement?
- Which user-facing workflow should I use for my study?
- Which exact API surface, schema, or class should I call in code?

Start with the overview if you need the package map, the fundamentals if you
need the technical grounding, or the guides if you want a runnable workflow
immediately.

What OpenPinch Covers
---------------------

- classical direct pinch targeting for process zones
- indirect, Total Process, and Total Site targeting through utility
  systems
- multi-utility studies and graph generation
- hierarchical zone modeling from unit operation to site scope
- integrated Heat Pump and refrigeration screening workflows
- above Pinch and below Pinch turbine cogeneration analysis
- Excel, JSON, CSV-bundle, CLI, notebook, and programmatic Python workflows

Recommended Reading Paths
-------------------------

New users
   Read :doc:`overview/what-is-openpinch`, then
   :doc:`guides/first-solve-cli` or :doc:`guides/first-solve-python`.

Process integration practitioners
   Read :doc:`fundamentals/pinch-analysis`,
   :doc:`fundamentals/direct-vs-indirect-integration`, and
   :doc:`fundamentals/graphs-and-interpretation` before diving into workflow
   guides.

Package integrators and advanced users
   Read :doc:`api/pinchproblem`, :doc:`api/service-layer`, and
   :doc:`api/schemas-and-config`, then use :doc:`api/generated-index` for the
   exhaustive package reference.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   overview/index
   fundamentals/index
   guides/index
   api/index
   examples/index
   developer/index
