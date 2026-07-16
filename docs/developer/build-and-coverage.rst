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

CI installs that generated wheel on Ubuntu, Windows, and macOS, then verifies
the package import, command-line help, and packaged resources without importing
the source checkout. Ubuntu remains the full-suite platform; Windows and macOS
provide core-runtime and wheel-install compatibility coverage.

Release Process
---------------

Production publication is tag-driven:

1. merge only after the required CI jobs pass
2. verify that ``pyproject.toml`` and ``uv.lock`` carry the intended version
3. run ``pytest -m solver`` in an environment with the required external solvers
4. create a signed or annotated ``vX.Y.Z`` tag at the intended commit
5. push the tag and approve the protected ``pypi`` environment after TestPyPI
   publication succeeds

The tag must exactly equal ``v{project.version}``. PR automation deliberately
uses ``--no-tag`` so maintainers retain explicit control of releases.

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

- CI runs Ruff, a warning-free docs build, and the non-solver suite with a 95%
  line-coverage floor.
- Every published optional extra, including ``synthesis``, has an isolated
  installation smoke check.
- Generated wheels are installed and smoke-tested on Ubuntu, Windows, and macOS.
- Docs consistency checks run under pytest as part of the normal suite.
- The docs build helper fails on Sphinx warnings by default.

Recommended Next Gates
----------------------

- keep packaged asset indexes synchronized with the resources module
- use link checking as an optional local audit, not a required CI or RTD gate,
  because external links can fail independently of documentation quality

Optional link audit:

.. code-block:: bash

   uv run python -m sphinx -b linkcheck docs docs/_build/linkcheck
