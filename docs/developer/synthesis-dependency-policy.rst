Synthesis Dependency Policy
===========================

OpenPinch core remains the default install target. The HEN synthesis migration
uses an isolated ``synthesis`` optional extra so users who only run core pinch,
targeting, graph-data, and workspace workflows do not install solver stacks,
plot export tooling, workbook engines, or wake-management packages.

Python Version Policy
---------------------

OpenPinch stays on Python ``>=3.14`` for the migration. Source OpenHENS targets
Python ``>=3.12``, but migrated synthesis dependencies must resolve and import
under OpenPinch's Python target before solver code moves.

The HENS-01 viability set is:

- ``pyomo>=6.10.0``
- ``gekko>=1.3.2``
- ``matplotlib>=3.10.9``
- ``plotly>=6.8.0``
- ``kaleido>=1.3.0``
- ``openpyxl>=3.1.5``
- ``wakepy>=1.0.0``

Optional Install
----------------

Install the future synthesis runtime dependencies explicitly:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"

Repository development should use uv:

.. code-block:: bash

   rtk uv sync --extra synthesis

The ``full`` extra intentionally does not include ``synthesis``. ``full`` keeps
covering the established dashboard, notebook, and Brayton-cycle optional
surfaces; HEN synthesis has separate solver-binary and dependency expectations.

Test Marker Policy
------------------

Fast tests are unmarked and must not require the ``synthesis`` extra or solver
binaries. Run them in default CI with:

.. code-block:: bash

   rtk uv run pytest -m "not synthesis and not solver"

Use ``@pytest.mark.synthesis`` for tests that require
``openpinch[synthesis]`` but no external solver binary:

.. code-block:: bash

   rtk uv run pytest -m synthesis

Use ``@pytest.mark.solver`` for tests that require external solver binaries
such as Couenne or IPOPT:

.. code-block:: bash

   rtk uv run pytest -m solver

Missing optional Python packages should raise
``MissingSynthesisDependencyError`` with the ``openpinch[synthesis]`` install
path. Missing external executables should raise ``MissingSynthesisSolverError``
with the binary name and the solver-marker rerun command.
