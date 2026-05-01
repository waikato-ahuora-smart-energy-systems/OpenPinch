Getting Started
===============

This guide covers the minimum path from installation to a working OpenPinch
analysis.

Requirements
------------

- Python ``>=3.14``
- ``numpy``, ``pandas``, and the dependencies listed in ``pyproject.toml``
- Optional: Jupyter if you want to use the packaged notebook series

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

Run A Sample Case
-----------------

OpenPinch ships with packaged sample cases so you can verify the workflow
without cloning the repository examples.

.. code-block:: bash

   openpinch sample -o basic_pinch.json
   openpinch validate basic_pinch.json
   openpinch run basic_pinch.json --graph-output graphs -o results

This will print a compact summary to the terminal, export an Excel workbook to
``results/``, and write graph HTML files to ``graphs/``.

Copy The Notebook Series
------------------------

To copy the packaged notebooks into your working directory:

.. code-block:: bash

   openpinch notebook -o notebooks

The notebook series is organized around distinct workflows, including basic
pinch analysis, graph interpretation, zonal analysis, heat-pump workflows, and
batch comparison.

Next Steps
----------

- Continue to :doc:`user-guide/quickstart` for the Python workflow.
- Use ``openpinch graph`` to export specific graph types directly from the CLI.
- Explore :doc:`reference/index` for the public API and architectural reference.
