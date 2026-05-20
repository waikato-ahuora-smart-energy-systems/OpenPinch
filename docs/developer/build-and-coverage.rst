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

Recommended Next Gates
----------------------

- fail the Sphinx build on warnings once all transient warning sources are removed
- add link checking
- keep packaged asset indexes synchronized with the resources module
