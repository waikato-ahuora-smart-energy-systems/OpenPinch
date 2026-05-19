What Is OpenPinch?
==================

OpenPinch is a process-integration toolkit for thermal targeting studies. It
combines classical pinch-analysis workflows with total-site style utility
integration, graph generation, and optional advanced workflows such as heat
pump integration screening and turbine cogeneration analysis.

The package is designed to support three working styles:

- command-line workflows for quick validation, run, and export tasks
- notebook and script workflows built around
  :class:`OpenPinch.PinchProblem` and :class:`OpenPinch.PinchWorkspace`
- programmatic service workflows built around validated schema payloads

Technical Scope
---------------

OpenPinch answers questions such as:

- What are the minimum hot and cold utility targets for this process?
- How much heat can be recovered internally before utilities are needed?
- How does a direct process-level answer differ from an indirect site-level
  answer?
- Which graph best explains the remaining utility load?
- Does a candidate heat-pump integration scenario improve the plant utility
  picture?
- How much above-pinch or below-pinch turbine work is theoretically available?

Core Product Shape
------------------

At a high level, the package turns validated inputs into a solved target set:

.. code-block:: text

   TargetInput / file input
           |
           v
     validation + preparation
           |
           v
        Zone hierarchy
           |
           v
     direct / indirect targeting
           |
           +--> graphs
           +--> summaries
           +--> Excel export
           +--> dashboard views

The same underlying analysis engine can be reached through:

- :class:`OpenPinch.PinchProblem`
- :class:`OpenPinch.PinchWorkspace`
- :func:`OpenPinch.main.pinch_analysis_service`
- the ``openpinch`` CLI
- packaged sample cases and notebooks

Who This Documentation Serves
-----------------------------

Thermal systems users
   Engineers and researchers who need the thermodynamic basis, workflow
   meaning, and output interpretation.

Python users
   Notebook and script users who need clear public entrypoints and example
   workflows.

Integrators and contributors
   Users embedding OpenPinch into larger software or extending the package
   internals.

What OpenPinch Does Not Assume
------------------------------

OpenPinch does not assume that every user wants the same depth of control.
Most users should stay at the `PinchProblem`, `PinchWorkspace`, or CLI level.
Advanced users can drop into schemas, service entrypoints, prepared `Zone`
trees, or lower-level analysis helpers when they need more control.

Next Steps
----------

- Use :doc:`capability-matrix` for the package feature map.
- Use :doc:`workflow-map` to choose the right entrypoint.
- Use :doc:`../fundamentals/index` for the technical grounding.
- Use :doc:`../guides/index` for runnable workflows.
