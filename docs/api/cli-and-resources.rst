CLI and Resources
=================

OpenPinch ships with a real command-line interface and packaged learning assets
in addition to the Python API. Those surfaces matter because much of the
package's practical power is meant to be discoverable without writing code from
scratch.

Command-Line Interface
----------------------

The CLI supports:

- ``run`` for end-to-end analysis and export
- ``graph`` for HTML graph export
- ``validate`` for payload preflight checks
- ``sample`` for copying packaged sample cases
- ``notebook`` for copying packaged example notebooks
- ``heat-pump`` for evaluating an integrated Heat Pump scenario against a case

.. automodule:: OpenPinch.__main__
   :members:
   :no-index:

Packaged Resources
------------------

The resources module exposes the packaged sample cases and notebooks used
throughout the guides and examples.

The packaged notebooks are intended to be copied as clean source assets. They
ship without stored execution output and rely on the same public
``PinchWorkspace`` and plotting surfaces documented elsewhere in RTD.

The current optional install split is:

- ``openpinch[notebook]`` for Jupyter, Plotly graph rendering, and Excel I/O
- ``openpinch[dashboard]`` for Streamlit plus the same plotting/export stack
- ``openpinch[brayton_cycle]`` for TESPy-backed Brayton-cycle tooling
- ``openpinch[full]`` for the combined optional surface

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

.. automodule:: streamlit_app
   :members:
   :no-index:

Where This Fits
---------------

Use the CLI and packaged resources when you want reproducible examples,
shareable graphs, or a fast onboarding path. Use the Python API pages when you
need integration into scripts, notebooks, or larger applications.
