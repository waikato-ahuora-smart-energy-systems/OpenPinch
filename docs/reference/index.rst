Reference Guide
===============

The reference pages provide exhaustive API documentation and build instructions
for contributors.

.. toctree::
   :maxdepth: 1

   api

.. _docs-build:

Building the Documentation
--------------------------

Install the documentation requirements and build the HTML pages locally with
``sphinx-build``:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs/ docs/_build/html

The Read the Docs configuration in ``.readthedocs.yaml`` uses the same command
and dependency set.
