Releasing
=========

OpenPinch has three GitHub Actions workflows:

``ci.yml``
   Validates every change. It runs on pull requests into ``develop`` and
   ``main``, on pushes to ``develop``, on demand, and inside ``release.yml``.
   Its final job is the only check branch rules need to require: it reports
   as **CI OK** for ``develop`` work and as **CI OK (main)** for pull requests
   into ``main`` and pushes to ``main``, which run the full profile.

``bump-version.yml``
   Runs after every push to ``develop``. If ``develop`` still carries the same
   version as ``main`` (the first merge after a release), it bumps the patch
   number and commits the change to ``develop``.

``release.yml``
   Runs on every push to ``main``. It runs ``ci.yml`` with the full profile
   and, when the project version is new, publishes the files that CI built.

Validation profiles
-------------------

.. list-table::
   :header-rows: 1

   * - Change
     - Lint, unit tests (95% branch coverage), docs, TESPy, performance, build, install smoke
     - Solver tests (Couenne, IPOPT)
     - Version-bump check
   * - PR into ``develop`` / push to ``develop``
     - yes
     - no
     - no
   * - PR into ``main``
     - yes
     - yes
     - yes
   * - Push to ``main`` (release)
     - yes
     - yes
     - no

Install smoke tests install the built wheel, not the checkout: the core
surface on Ubuntu, Windows and macOS, and every optional extra on Ubuntu.
Tests use Hypothesis seed ``20260715``. Python comes from ``.python-version``.

Cutting a release
-----------------

1. Merge pull requests into ``develop`` as usual. The first merge after a
   release makes ``bump-version.yml`` commit the next patch version
   (for example 0.6.10 → 0.6.11) to ``pyproject.toml``, ``uv.lock`` and
   ``.bumpversion.toml``. Later merges keep that version. Pull before you
   continue working on ``develop``.
2. For a minor or major release instead, bump on ``develop`` yourself; the
   workflow leaves a version that is already ahead of ``main`` alone:

   .. code-block:: bash

      uvx bump-my-version bump minor   # or major

3. Open the ``develop`` → ``main`` pull request. Its **Version bump** job
   fails unless the version is higher than ``main``'s and has no
   ``v<version>`` tag.
4. Merge once **CI OK (main)** is green. ``release.yml`` then re-validates the
   merged commit, uploads the wheel and sdist to PyPI with trusted publishing,
   and creates the ``v<version>`` tag and GitHub release with the same files
   attached.

The bump commit is pushed with the workflow token, which does not start new
workflow runs. If a ``develop`` → ``main`` pull request is already open when
the bump lands, close and reopen it (or push a commit) so its checks run on
the bumped commit.

A push to ``main`` whose version is already tagged runs CI only and publishes
nothing.

When a release fails
--------------------

Open the failed **Release** run and choose **Re-run failed jobs** (or re-run
the whole workflow). Every step is safe to repeat: the PyPI upload skips files
that already exist, and a missing, draft or incomplete GitHub release is
created or completed. A version only counts as released once its published
GitHub release carries both files, so a half-finished release is resumed rather
than skipped. Builds are reproducible (``SOURCE_DATE_EPOCH`` is the commit
time), so a full re-run produces the same bytes. A version that reached PyPI can
never be replaced; if its files are wrong, bump the version and release again.

Release planning on ``main`` refuses to publish, and fails the run, when:

- the version is lower than an existing ``vX.Y.Z`` tag (for example, a stale
  pull request merged after a newer release), or
- ``v<version>`` already exists on a different commit.

Bump the version above the latest release and merge again.

One-time repository settings
----------------------------

- Branch rule for ``develop``: require **CI OK**, and allow GitHub Actions to
  push the version-bump commit (``develop`` is currently unprotected, which
  works as is).
- Branch rule for ``main``: require **CI OK (main)**. Remove the old required
  checks (``OpenPinch PR Gate``, ``test``).
- PyPI trusted publisher for ``OpenPinch``: set the workflow file to
  ``release.yml`` and the environment to ``pypi``. The old ``ci-publish.yml``
  publisher and the TestPyPI publisher are no longer used.
- Environment ``pypi``: restrict deployments to the ``main`` branch.
- A pull request retargeted from ``develop`` to ``main`` has no
  **CI OK (main)** check until its checks re-run, so it cannot merge early:
  push a commit or close and reopen it. Title and description edits do not
  re-run CI.
