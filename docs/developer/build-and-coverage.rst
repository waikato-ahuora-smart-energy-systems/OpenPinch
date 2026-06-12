Build and Coverage
==================

The documentation should be maintained like a tested product surface, not a
side artifact. This page records the local build workflow and the expected
quality bar.

Local Build
-----------

Build the HTML documentation from the repository root:

.. code-block:: bash

   uv run scripts/build_docs.py

The generated site is written to ``docs/_build/html``.

The helper runs Sphinx with ``--fail-on-warning --keep-going`` so stale
``automodule`` paths, broken cross references, and other warning-level RTD
problems fail before publication.

Release Build
-------------

Build the wheel and source distribution from the repository root:

.. code-block:: bash

   uv run scripts/build_dist.py

Alternative Direct Sphinx Build
-------------------------------

If you need to run Sphinx directly:

.. code-block:: bash

   uv run python -m sphinx -b html docs docs/_build/html

Use the stricter form when checking a documentation change:

.. code-block:: bash

   uv run python -m sphinx -b html --fail-on-warning --keep-going docs docs/_build/html

Coverage Expectations
---------------------

The target state for docs coverage is:

- every stable package-root export documented in the curated API pages
- the full ``PinchProblem`` workflow documented in both guides and reference
- every packaged sample case and notebook represented in the examples section
- support status called out explicitly for partial or expert-only subsystems

Current Quality Gates
---------------------

- CI now runs a docs HTML build alongside the main test workflow.
- Docs consistency checks run under pytest as part of the normal suite.
- The docs build helper fails on Sphinx warnings by default.

Recommended Next Gates
----------------------

- add link checking
- keep packaged asset indexes synchronized with the resources module
