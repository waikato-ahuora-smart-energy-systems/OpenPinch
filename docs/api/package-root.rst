Package Root
============

The :mod:`OpenPinch` package root is the primary import surface for notebooks,
small scripts, and lightweight applications. It intentionally re-exports a
small set of high-value entry points so most users do not need to navigate the
full module tree.

Recommended Imports
-------------------

Use the package root when you want the supported front door:

- :class:`OpenPinch.PinchProblem` for single-case file-backed or notebook-driven workflows
- :class:`OpenPinch.PinchWorkspace` for named multi-case studies and notebook workflows
- :func:`OpenPinch.pinch_analysis_service` for typed programmatic execution
- :func:`OpenPinch.get_piecewise_linearisation_for_streams` for nonlinear
  stream segmentation utilities
- :mod:`OpenPinch.lib` re-exports when you are constructing typed inputs with
  enums, schemas, or configuration objects

Typical Pattern
---------------

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace, pinch_analysis_service
   from OpenPinch.lib.schemas.io import TargetInput

   problem = PinchProblem("basic_pinch.json")
   result = problem.target()

   workspace = PinchWorkspace(
       source="crude_preheat_train.json",
       project_name="crude_preheat_train",
   )
   workspace.copy_case("baseline", "wide_dt", activate=False)
   workspace.set_dt_cont_multiplier(0.5, case_name="wide_dt")
   comparison = workspace.compare_cases("baseline", "wide_dt")

The package root is intentionally small. If your workflow starts depending on
prepared zones, direct service orchestration, or lower-level targeting
algorithms, move down into the pages under :doc:`service-layer` and
:doc:`domain-model`.

It is also a Python-first surface. The package root does not hide the fact that
real solves, comparisons, exports, and advanced targeting happen through Python
objects even though the project also ships packaged learning notebooks.

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

Use :doc:`schemas-and-config` when you need the typed input/config layer,
and :doc:`generated-index` when you need exhaustive module-level details.
