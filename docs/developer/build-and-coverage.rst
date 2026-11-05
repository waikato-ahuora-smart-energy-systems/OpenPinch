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
``automodule`` paths, broken cross references, and other warning-level Read the
Docs problems fail before publication. This is the command to run before
pushing a documentation change; the docs lane in CI runs the same build.

Release Build
-------------

Build the wheel and source distribution from the repository root:

.. code-block:: bash

   uv run scripts/build_dist.py

CI builds the wheel with ``uv build --out-dir dist`` (the same build backend),
installs it on Ubuntu, Windows, and macOS, then verifies
the package import, command-line help, and packaged resources without importing
the source checkout. Ubuntu remains the full-suite platform; Windows and macOS
provide core-runtime and wheel-install compatibility coverage.

CI Lanes
--------

``ci.yml`` owns branch coverage, docs, TESPy, optional-install and
cross-platform wheel smoke tests, and the Pyodide lane, which is required for
``CI OK``. Pull requests into ``main`` and every push to ``main`` also run the
solver tests, the performance lane, the py312 and py313 unit lanes and the
``floors`` lane (runtime dependencies installed at the lowest versions
``pyproject.toml`` allows). Branch coverage must stay at or above 95 percent.
The unit lane runs on parallel test workers, and tests run optimisation tasks
serially unless marked ``process_pools``. The release lanes keep real worker
processes for every test by setting ``OPENPINCH_TEST_REAL_POOLS=1``. ``release.yml`` publishes from ``main`` when the version is new; for
how to cut a release, see :doc:`releasing`.

Repository Controls
-------------------

Require the ``CI OK`` status check on ``develop`` and ``CI OK (main)`` on
``main``. Each is the single gate that summarises every other CI job; the
``(main)`` variant is only reported by full-profile runs.

Direct Sphinx Build
-------------------

The helper above is the required check. To debug a build, run Sphinx directly:

.. code-block:: bash

   uv run python -m sphinx -b html docs docs/_build/html

The stricter form matches what the helper runs:

.. code-block:: bash

   uv run python -m sphinx -b html --fail-on-warning --keep-going docs docs/_build/html

Probity
-------

Probity (``@nizos/probity``, configured in ``probity.config.ts``) is a Claude
Code ``PreToolUse`` hook (``.claude/settings.json``), so it applies to agent
sessions, not to plain terminal git. It blocks a commit until a passing
``pytest`` run follows the last file edit, enforces test-first edits under
``OpenPinch/`` and ``tests/``, and rejects force pushes, skipped or xfailed
tests, lint or type suppression comments, new lint ignores in configuration,
and emojis in Markdown. When a commit is refused, run the pytest files that
cover the change and retry. Once per clone, run ``nvm use && npm install`` and
log in with ``claude /login`` so the test-first validator can run.

Coverage Expectations
---------------------

The target state for docs coverage is:

- the sole main-service contract documented in the curated API pages
- internal ``PinchProblem`` workflows explicitly labelled unsupported
- every packaged sample case and notebook represented in the examples section
- support status called out explicitly for partial or expert-only subsystems

Current Quality Gates
---------------------

- CI runs Ruff, a warning-free docs build, and the unit suite (everything except
  solver, TESPy, performance, and docs tests) with a 95% branch-aware coverage
  floor.
- Every published optional extra, including ``synthesis`` and the aggregate
  ``full``, has an isolated installation smoke check.
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
