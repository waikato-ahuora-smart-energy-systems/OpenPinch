Reference Guide
===============

The reference pages provide architecture context, complete API documentation,
and local build instructions.

.. toctree::
   :maxdepth: 1

   architecture
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
