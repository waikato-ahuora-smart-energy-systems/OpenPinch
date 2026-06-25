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
- resource helpers such as ``list_sample_cases()``, ``sample_case_metadata()``,
  ``list_notebooks()``, and ``copy_notebook()``
- report helpers such as ``problem.validation_report()``, ``problem.report()``,
  and ``problem.metrics()``

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
   workspace.scenario("wide_dt", dt_cont_multiplier=0.5)
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

Package-Level Import Notes
--------------------------

The root package keeps a curated set of commonly used classes, schemas, enums,
resource helpers, and report types. The broader :mod:`OpenPinch.lib` package is
still importable for advanced schema and enum work, but it is not part of the
root wildcard export surface.

Use :doc:`schemas-and-config` when you need the typed input/config layer,
and :doc:`generated-index` when you need exhaustive module-level details.
