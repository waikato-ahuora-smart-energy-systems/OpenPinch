Package Root
============

The :mod:`OpenPinch` package root is the primary import surface for notebooks,
small scripts, and lightweight applications. It intentionally re-exports a
small set of high-value entry points so most users do not need to navigate the
full module tree.

Recommended Imports
-------------------

Use the package root when you want the supported front door:

- :class:`OpenPinch.PinchProblem` for file-backed or notebook-driven workflows
- :func:`OpenPinch.pinch_analysis_service` for typed programmatic execution
- :func:`OpenPinch.get_piecewise_linearisation_for_streams` for nonlinear
  stream segmentation utilities
- :mod:`OpenPinch.lib` re-exports when you are constructing payloads with
  enums, schemas, or configuration objects

Typical Pattern
---------------

.. code-block:: python

   from OpenPinch import PinchProblem, pinch_analysis_service
   from OpenPinch.lib.schemas.io import TargetInput

   problem = PinchProblem("basic_pinch.json")
   result = problem.target()

The package root is intentionally small. If your workflow starts depending on
prepared zones, direct service orchestration, or lower-level targeting
algorithms, move down into the pages under :doc:`service-layer` and
:doc:`domain-model`.

Root Exports
------------

.. automodule:: OpenPinch
   :no-members:
   :no-index:

High-Level Service Function
---------------------------

.. autofunction:: OpenPinch.main.pinch_analysis_service
   :no-index:

Package-Level Re-Export Notes
-----------------------------

The root package also re-exports the contents of :mod:`OpenPinch.lib`. That is
useful for interactive work, but for larger codebases it is usually clearer to
import schemas, enums, and configuration types from their explicit modules.

Use :doc:`schemas-and-config` when you need the typed payload and config layer,
and :doc:`generated-index` when you need exhaustive module-level details.
