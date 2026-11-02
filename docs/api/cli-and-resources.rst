CLI and Resources
=================

OpenPinch ships packaged learning assets in addition to the Python API. The
resource helpers are the primary programmatic surface for discovering,
inspecting, and copying those assets. The command-line surface is intentionally
narrow: it copies notebooks and writes contract JSON Schemas.

Command-Line Interface
----------------------

The published package currently exposes two CLI subcommands:

- ``notebook`` for copying packaged example notebooks
- ``schema`` for writing the JSON Schema of each public contract to
  ``<Contract>.schema.json`` files

.. automodule:: OpenPinch.__main__
   :members:
   :no-index:

This means the CLI is an onboarding and schema-export surface, not a solve
surface. Validation,
targeting, graph export, Excel export, and dashboard launch all happen through
Python.

Packaged Resources
------------------

The resources module exposes the packaged sample cases
and notebooks used throughout the guides and examples.

The main helpers are:

- ``list_sample_cases()``, ``sample_case_metadata()``, and
  ``read_sample_case()`` for discovery and inspection
- ``copy_sample_case()`` for local editable copies
- ``list_notebooks()``, ``notebook_metadata()``, and ``copy_notebook()`` for
  the packaged notebook series

The ``PinchProblem`` and ``PinchWorkspace`` classes also resolve
packaged sample-case names such as ``basic_pinch.json`` directly when no local
file with the same name exists.

The packaged notebooks are saved without execution outputs and can be copied
for local exploration. They exercise the ``PinchProblem`` and
``PinchWorkspace`` workflows. The series is ordered around first solves, Total
Site method interpretation, workspace cases and persistence, multiperiod
studies, area, cost and exergy, heat pump and refrigeration screening, process
MVR and cascade mechanics, cogeneration, energy-transfer analysis, HEN
synthesis, results, plots and exports, and utility placement.

The current optional install split is:

- ``openpinch[notebook]`` for Jupyter, Plotly graph rendering, and Excel I/O
- ``openpinch[dashboard]`` for Streamlit plus the same plotting/export stack
- ``openpinch[synthesis]`` for solver-backed heat exchanger network synthesis
- ``openpinch[brayton_cycle]`` as an alias of ``openpinch[tespy]``; the public
  Brayton targeting methods currently raise ``NotImplementedError``
- ``openpinch[tespy]`` for optional TESPy vapour-compression targeting and
  target-owned HPR performance-map generation
- ``openpinch[full]`` for every optional surface, including the IDAES/Pyomo
  synthesis stack; run ``idaes get-extensions`` after installation

.. automodule:: OpenPinch.resources
   :members:
   :no-index:

Packaged Asset Modules
----------------------

Tutorial notebooks and sample cases live in ``OpenPinch/tutorials/notebooks``
and ``OpenPinch/tutorials/sample_cases``. HPR interchange schemas and canonical
examples live in ``OpenPinch/contracts/resources/hpr_performance_map``. Runtime
compressor characteristics live in
``OpenPinch/analysis/heat_pumps/performance_maps/characteristics``. Resource helper
calls and sample-case name resolution are unchanged by this layout.

.. automodule:: OpenPinch.tutorials
   :no-members:
   :no-index:

.. automodule:: OpenPinch.contracts.resources
   :no-members:
   :no-index:

.. automodule:: OpenPinch.tutorials.sample_cases
   :no-members:
   :no-index:

.. automodule:: OpenPinch.tutorials.notebooks
   :no-members:
   :no-index:

Dashboard Surface
-----------------

OpenPinch also includes a Streamlit-oriented graphing and dashboard path for
interactive exploration after solving a problem. Install
``openpinch[dashboard]`` before using this surface.

The repository-level ``streamlit_app.py`` module is documented here as a local
demo entrypoint for contributors. It is not part of the published wheel.
Application code can call ``problem.show_dashboard()`` explicitly after
targeting.

.. automodule:: streamlit_app
   :members:
   :no-index:

Where This Fits
---------------

Use packaged resources when you want reproducible examples, shareable learning
assets, or a fast onboarding path. Use :doc:`package-root` for the application
contract; use the other API pages only when contributor-level detail is
acceptable.
