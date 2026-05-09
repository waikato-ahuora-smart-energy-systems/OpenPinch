Reference Guide
===============

The reference pages provide architecture context, complete API documentation,
and local build instructions.

Use :doc:`architecture` to understand how zones, problem tables, and targeting
stages fit together. Use :doc:`api` when you need callable signatures,
configuration objects, schema definitions, or lower-level analysis modules.

.. toctree::
   :maxdepth: 1

   architecture
   api

.. _docs-build:

Building the Documentation
--------------------------

Build the HTML pages locally with ``uv``:

.. code-block:: bash

   uv run scripts/build_docs.py

The helper script installs the local Sphinx requirements declared in
``scripts/build_docs.py`` and writes the output to ``docs/_build/html``.
Read the Docs uses the environment defined in ``.readthedocs.yaml`` and builds
the same ``docs/conf.py`` configuration.
