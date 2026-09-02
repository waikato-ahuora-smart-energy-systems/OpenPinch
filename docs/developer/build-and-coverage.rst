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
2. merge only after the read-only validation jobs, external-solver suite, and
   aggregate ``pr-gate`` result pass
3. let the main-branch workflow repeat the test, documentation, solver, build,
   and cross-platform artifact gates
4. after validation, the workflow creates the annotated version tag and a
   draft GitHub release containing checksummed release artifacts, then
   publishes the same distributions to TestPyPI; exact filename and SHA-256
   preflight/postflight checks safely distinguish absent, partial, complete,
   and mismatched index state
5. after TestPyPI succeeds:

   * it publishes the GitHub release before production PyPI
   * it dispatches the same workflow at the version tag
6. the tag-ref run verifies the source push, workflow identity, latest required
   jobs, and immutable build artifact ID, digest, and build attempt; it then
   requires the public release files to match that artifact byte-for-byte
   without rebuilding and waits at the protected ``pypi`` environment
7. after approval, PyPI Trusted Publishing uploads the verified distributions
   and a separate unprivileged job confirms the version through the PyPI API

Version bumping does not create a local tag. The release workflow owns the
``v{project.version}`` tag and rejects malformed versions, lock mismatches, or
an existing tag that points to a different commit. A production failure can
therefore leave a public GitHub Release while PyPI is pending. Open the
original tag run and select **Re-run failed jobs**. Exact index preflight,
``skip-existing``, and a separately retryable availability check recover an
absent, partial, or already-complete release without accepting mismatched
files. Do not start a fresh tag dispatch when the upload may already have
succeeded.

Repository Controls
-------------------

Keep the repository controls aligned with the workflow contract:

* require full-length commit SHAs for Actions and keep the default
  ``GITHUB_TOKEN`` read-only; grant write scopes only on the release jobs that
  need them
* do not let GitHub Actions create or approve pull requests
* require pull requests, conversation resolution, an up-to-date branch, and
  the unique ``pr-gate`` check before merging ``main``; do not require a Code
  Owner review unless the repository contains a maintained ``CODEOWNERS`` file
* protect stable ``v*.*.*`` tags from updates and deletion while leaving tag
  creation available to the release workflow
* retain a reviewer on the protected ``pypi`` environment; a sole-maintainer
  repository cannot also require self-review prevention without introducing a
  second eligible reviewer

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
