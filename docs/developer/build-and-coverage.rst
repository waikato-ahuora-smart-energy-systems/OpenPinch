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

Production publication is automated from the ``main`` branch:

1. set a strict, forward ``X.Y.Z`` version in ``pyproject.toml`` and keep the
   OpenPinch entry in ``uv.lock`` synchronized in the pull request
2. merge only after the required read-only validation jobs pass
3. let the main-branch workflow repeat the test, documentation, solver, build,
   and cross-platform artifact gates
4. after validation, the workflow creates the annotated version tag and a
   draft GitHub release containing checksummed release artifacts, then
   publishes the same distributions to TestPyPI
5. after TestPyPI succeeds:

   * it publishes the GitHub release before production PyPI
   * it dispatches the same workflow at the version tag
6. the tag-ref run downloads the public release artifacts, verifies their
   names and SHA-256 digests without rebuilding, and then waits at the
   protected ``pypi`` environment
7. after approval, PyPI Trusted Publishing uploads the verified distributions
   and the workflow confirms the version through the PyPI API

Version bumping does not create a local tag. The release workflow owns the
``v{project.version}`` tag and rejects malformed versions, lock mismatches, or
an existing tag that points to a different commit. A production failure can
therefore leave a public GitHub Release while PyPI is pending. Rerun
``ci-publish.yml`` at that version tag with the matching ``release_tag`` input;
the tag phase reuses the published artifacts and does not rebuild the release.

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

- the sole main-service contract documented in the curated API pages
- internal ``PinchProblem`` workflows explicitly labelled unsupported
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
