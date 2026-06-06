Support and Stability
=====================

Not every exposed OpenPinch surface should be read the same way. This page
defines the support levels used throughout the documentation.

Support Levels
--------------

Stable public API
   Preferred user-facing surfaces documented as the main package entrypoints.

Advanced public API
   Supported surfaces intended for expert workflows, deeper post-processing, or
   greater control over the targeting pipeline.

Experimental / partial
   Exposed modules or concepts that are present in the package but are not yet
   documented or polished to the same standard as the core workflow.

Internal implementation
   Package internals that may be useful for understanding the codebase but
   should not be treated as a stable extension surface by default.

Stable Public Surfaces
----------------------

- :class:`OpenPinch.PinchProblem`
- :class:`OpenPinch.PinchWorkspace`
- :func:`OpenPinch.main.pinch_analysis_service`
- root package imports from :mod:`OpenPinch`
- `TargetInput`, `TargetOutput`, and the main I/O schema models
- the ``openpinch notebook`` CLI command
- packaged sample-case and notebook resource helpers

Advanced Public Surfaces
------------------------

- ``problem.target.*`` targeting accessors
- ``problem.plot.*`` graph accessors
- exergy post-processing through ``problem.target.exergy(...)`` and the
  matching exergetic plot accessors
- :func:`OpenPinch.services.input_data_processing.data_preparation.prepare_problem`
- service-layer targeting helpers under :mod:`OpenPinch.services`
- domain classes such as `Zone`, `Stream`, `StreamCollection`, and
  `ProblemTable`
- stream linearisation and some other utility helpers

Experimental or Partial Surfaces
--------------------------------

- community and region framing as a user-facing multiscale workflow
- energy transfer analysis modules
- lower-level exergy helper modules below the accessor surface
- some lower-level HPR comparison concepts that sit below the explicit
  ``problem.target.*`` targeting workflows

These surfaces may still be valuable, but the docs treat them with more
qualification than the core package workflows.

Internal or Cautionary Surfaces
-------------------------------

- helper functions inside `PinchProblem` and lower-level service modules that
  exist mainly to support public entrypoints
- low-level optimizer backends
- specialist cycle implementation internals

Documentation Rule
------------------

When reading the API reference:

- start with the curated pages in :doc:`../api/index`
- use the generated appendix only after you know which layer you want
- assume undocumented helpers are internal unless explicitly called out here or
  in the curated API pages

Implications For Contributors
-----------------------------

Changes to stable public surfaces should trigger:

- docs updates
- example updates where relevant
- tests for user-facing behavior

Changes to advanced, experimental, or internal surfaces still matter, but they
should be documented in proportion to their intended user visibility.
