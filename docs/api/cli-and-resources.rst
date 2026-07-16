CLI and Resources
=================

OpenPinch ships packaged learning assets in addition to the Python API. The
resource helpers are the primary programmatic surface for discovering,
inspecting, and copying those assets. The command-line surface is intentionally
narrow and only copies notebooks.

Command-Line Interface
----------------------

The published package currently exposes one CLI subcommand:

- ``notebook`` for copying packaged example notebooks

.. automodule:: OpenPinch.__main__
   :members:
   :no-index:

This means the CLI is an onboarding surface, not a solve surface. Validation,
targeting, graph export, Excel export, and dashboard launch all happen through
Python.

Packaged Resources
------------------

The resources module exposes the packaged sample cases and notebooks used
throughout the guides and examples.

The main helpers are:

- ``list_sample_cases()``, ``sample_case_metadata()``, and
  ``read_sample_case()`` for discovery and inspection
- ``copy_sample_case()`` for local editable copies
- ``list_notebooks()``, ``notebook_metadata()``, and ``copy_notebook()`` for
  the packaged notebook series

``PinchProblem`` and ``PinchWorkspace`` also resolve packaged sample-case names
such as ``basic_pinch.json`` directly when no local file with the same name
exists.

The packaged notebooks are intended to be copied as clean source assets. They
ship without stored execution output and rely on the same public
``PinchWorkspace`` and plotting surfaces documented elsewhere in RTD. The
series is ordered around first solves, Total Site method interpretation,
multiperiod studies, advanced Heat Pump screening, direct gas/vapour MVR,
VC+MVR cascade mechanics, HEN synthesis, energy-transfer analysis, and
schema/service integration.

The current optional install split is:

- ``openpinch[notebook]`` for Jupyter, Plotly graph rendering, and Excel I/O
- ``openpinch[dashboard]`` for Streamlit plus the same plotting/export stack
- ``openpinch[synthesis]`` for solver-backed heat exchanger network synthesis
- ``openpinch[brayton_cycle]`` for TESPy-backed Brayton-cycle tooling
- ``openpinch[full]`` for every optional surface, including the IDAES/Pyomo
  synthesis stack; run ``idaes get-extensions`` after installation

.. automodule:: OpenPinch.resources
   :members:
   :no-index:

Packaged Asset Modules
----------------------

.. automodule:: OpenPinch.data
   :no-members:
   :no-index:

.. automodule:: OpenPinch.data.sample_cases
   :no-members:
   :no-index:

.. automodule:: OpenPinch.data.notebooks
   :no-members:
   :no-index:

Dashboard Surface
-----------------

OpenPinch also includes a Streamlit-oriented graphing and dashboard path for
interactive exploration after solving a problem. Install
``openpinch[dashboard]`` before using this surface.

The repository-level ``streamlit_app.py`` module is documented here as a local
demo entrypoint for contributors. It is not part of the published wheel. For
installed package usage, prefer :meth:`OpenPinch.PinchProblem.show_dashboard`
or :func:`OpenPinch.streamlit_webviewer.web_graphing.render_streamlit_dashboard`.

.. automodule:: streamlit_app
   :members:
   :no-index:

Where This Fits
---------------

Use packaged resources when you want reproducible examples, shareable learning
assets, or a fast onboarding path. Use the Python API pages when you need
integration into scripts, notebooks, or larger applications.
