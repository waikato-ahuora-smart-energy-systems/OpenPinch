Support and Stability
=====================

OpenPinch documents different surfaces with different promises. Use this page
when deciding whether a callable is a stable user interface, an advanced
workflow, or an implementation detail.

Stable Public Surfaces
----------------------

Stable surfaces are the preferred way to build user workflows:

- ``PinchProblem``
- ``PinchWorkspace``
- ``pinch_analysis_service``
- root package imports listed in :mod:`OpenPinch`
- ``TargetInput``, ``TargetOutput``, and the main I/O schema models
- packaged resource helpers such as ``list_sample_cases()`` and
  ``copy_notebook()``
- ``openpinch notebook`` for copying notebooks only
- ``validation_report()``, ``summary_frame()``, ``report()``, ``metrics()``,
  and ``config_options()``

Advanced Public Surfaces
------------------------

Advanced surfaces are supported, but they require more interpretation:

- ``problem.target.*`` specialized targeting accessors
- ``problem.plot.*`` graph accessors
- ``problem.add_component.process_mvr(...)``
- ``problem.design.*`` heat exchanger network synthesis accessors
- exergy and cogeneration post-processing
- service-layer helpers under :mod:`OpenPinch.services`
- domain classes such as ``Zone``, ``Stream``, ``StreamCollection``, and
  ``ProblemTable``

Experimental or Partial Surfaces
--------------------------------

These surfaces may be useful, but they are not presented as primary user
workflows:

- community and region framing
- lower-level energy-transfer and exergy helper modules
- optimizer backend internals
- implementation modules below the curated service/API pages

Dependency Boundaries
---------------------

Optional dependencies are intentionally workflow-specific:

- ``openpinch[notebook]`` for Jupyter, Plotly graph rendering, and Excel I/O
- ``openpinch[dashboard]`` for Streamlit dashboard review
- ``openpinch[synthesis]`` plus solver extensions for solver-backed HEN synthesis
- ``openpinch[brayton_cycle]`` for TESPy-backed Brayton-cycle tooling

Documentation Rule
------------------

When a stable public surface changes, update:

1. the task guide that teaches the workflow
2. the curated API page that documents the contract
3. examples or packaged asset docs when the change affects learning workflows
4. docs consistency tests

When an advanced or experimental surface changes, document it in proportion to
its intended user visibility and mark the support level clearly.

Next Steps
----------

- :doc:`capability-matrix` for feature status by workflow.
- :doc:`../developer/docs-conventions` for contributor standards.
