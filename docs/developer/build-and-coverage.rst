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

Normal merges to ``main`` validate without publishing or changing versions.
PR validation is read-only. Explicit release preparation, publication, and
same-artifact recovery are described in :doc:`releasing`.

The shared validator owns ordinary branch coverage, docs, TESPy, performance,
optional installs, and cross-platform artifact smoke. Main PRs, main pushes,
and new releases additionally require solver validation. Ordinary coverage
remains 95 percent. Exact compatible develop evidence may replace integration
lanes; it never replaces main's solver requirement.

Repository Controls
-------------------

Require ``OpenPinch PR Gate`` after its hosted check name has been verified.
The temporary ``test`` compatibility check also requires the complete gate;
remove it only after the new required-check configuration is active. Preserve
existing review, strictness, and tag protections during migration. Remote
settings changes require separate authorization and verification.

PR description-only edits do not run expensive validation. They do not create
new success evidence: reopen a ready PR or push a candidate change if a new
complete gate is required. Base-branch edits trigger the appropriate profile.

Retain the protected ``pypi`` environment and trusted-publisher configuration.
Before activating explicit main-context publication, verify allowed refs and
reviewer settings without weakening them to make a run pass.

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
  branch-aware coverage floor.
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
