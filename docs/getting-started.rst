Getting Started
===============

This guide covers the minimum you need to install OpenPinch, validate data, and
invoke the analysis engine.

Requirements
------------

- Python 3.11 or newer (3.12 is fully supported)
- ``numpy``, ``pandas``, and other dependencies listed in ``pyproject.toml``
- Optional: Microsoft Excel if you plan to export results to workbooks

Installation
------------

Install the published package from PyPI:

.. code-block:: bash

   python -m pip install openpinch

To work on the project locally, clone the repository and install it in editable
mode:

.. code-block:: bash

   git clone https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch.git
   cd OpenPinch
   python -m pip install -e .
   python -m pip install -r docs/requirements.txt

Quick Health Check
------------------

Once installed, verify that the package imports and the command-line entry point
responds:

.. code-block:: bash

   python -c "import importlib.metadata; print(importlib.metadata.version('OpenPinch'))"
   python -m OpenPinch --help

If you intend to build the documentation locally, install the dependencies under
``docs/requirements.txt`` and run ``sphinx-build`` as described in
:ref:`docs-build`.
