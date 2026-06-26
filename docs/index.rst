OpenPinch Documentation
=======================

OpenPinch is a Python toolkit for Pinch Analysis, Total Site heat integration,
graph interpretation, Heat Pump and refrigeration screening, exergy
post-processing, cogeneration targeting, and heat exchanger network synthesis.
It is built for engineers and researchers who need reproducible thermal
targeting workflows in scripts, notebooks, and applications.

OpenPinch exposes one analysis engine through three public front doors:

- :class:`OpenPinch.PinchProblem` for one case at a time
- :class:`OpenPinch.PinchWorkspace` for named multi-case studies and bundle
  persistence
- :func:`OpenPinch.main.pinch_analysis_service` for typed request/response
  integration

The published CLI is intentionally narrow: ``openpinch notebook`` copies the
packaged notebook series, while solving, validation, graph export, Excel
export, and advanced targeting all happen in Python.

Start Here
----------

I want to solve a case
   Start with :doc:`getting-started`, then use
   :doc:`guides/first-solve-python` for the main Python workflow or
   :doc:`guides/notebooks-and-sample-cases` for packaged examples.

I need to understand the method
   Start with :doc:`overview/what-is-openpinch`, then read
   :doc:`fundamentals/pinch-analysis`,
   :doc:`fundamentals/direct-vs-indirect-integration`, and
   :doc:`fundamentals/graphs-and-interpretation`.

I am integrating or extending OpenPinch
   Start with :doc:`api/pinchproblem`, :doc:`api/schemas-and-config`, and
   :doc:`api/service-layer`; use :doc:`api/generated-index` only after you know
   which layer you need.

What OpenPinch Covers
---------------------

- direct process Pinch Analysis and indirect Total Site targeting
- hierarchical zone modeling from unit operation to site scale
- Composite Curve, Grand Composite Curve, Total Site profile, and SUGCC graphs
- JSON, Excel, CSV-bundle, schema-first, and packaged sample-case inputs
- Heat Pump and refrigeration screening, including simulated-cycle backends
- direct gas/vapour MVR process-component studies
- exergy and cogeneration post-processing on solved thermal targets
- heat exchanger network synthesis through the ``problem.design`` accessor

The documentation is organized as a manual first and an API reference second:
overview pages help you choose a workflow, fundamentals explain the method,
guides provide runnable tasks, examples map packaged assets to decisions, and
API pages document the exact public contracts.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting-started
   overview/index
   fundamentals/index
   guides/index
   api/index
   examples/index
   developer/index
