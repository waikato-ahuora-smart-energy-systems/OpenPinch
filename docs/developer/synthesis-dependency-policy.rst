Synthesis Dependency Policy
===========================

OpenPinch core remains the default install target. Heat exchanger network
synthesis uses an isolated ``synthesis`` optional extra so users who
only run core pinch, targeting, graph-data, and workspace workflows do not
install solver stacks, plot export tooling, workbook engines, or
wake-management packages.

Python Version Policy
---------------------

OpenPinch supports Python ``>=3.12``, the same floor as the OpenHENS source, and
develops on the version in ``.python-version``. Synthesis dependencies must
resolve and import on every supported Python; CI runs the unit tests on the
development version for every change, and on 3.12 and 3.13 for pull requests
into and pushes to ``main``.

The synthesis dependency set is:

- ``pyomo>=6.10.0``
- ``gekko>=1.3.2``
- ``plotly>=6.8.0``
- ``kaleido>=1.3.0``
- ``openpyxl>=3.1.5``
- ``wakepy>=1.0.0``
- ``idaes-pse>=2.11.0``
- ``highspy>=1.11.0``

Optional Install
----------------

Install the synthesis runtime dependencies explicitly:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"

Repository development uses the locked ``dev`` group, which already includes
the synthesis stack:

.. code-block:: bash

   uv sync --frozen --group dev

The ``full`` extra includes the synthesis stack (Pyomo, GEKKO, Kaleido, wakepy,
IDAES and HiGHS) together with the dashboard, notebook, TESPy, and Brayton-cycle
surfaces. External solver binaries are still installed separately.

Test Marker Policy
------------------

Fast tests are unmarked and should not require the ``synthesis`` extra or solver
binaries. CI does not enforce this: the unit lanes sync the ``dev`` group,
which already contains the synthesis stack, and select tests with
``-m "not solver and not tespy and not performance and not docs"``, so
``synthesis``-marked tests run there. To run only the fast tests locally:

.. code-block:: bash

   uv run --no-sync pytest -m "not synthesis and not solver"

Use ``@pytest.mark.synthesis`` for tests that require
``openpinch[synthesis]`` but no external solver binary:

.. code-block:: bash

   uv run --no-sync pytest -m synthesis

Use ``@pytest.mark.solver`` for tests that require external solver binaries
such as Couenne or IPOPT:

.. code-block:: bash

   uv run --no-sync pytest -m solver

Missing optional Python packages should raise
``MissingSynthesisDependencyError`` with the ``openpinch[synthesis]`` install
path. Missing external executables should raise ``MissingSynthesisSolverError``
with the binary name and a pointer to these policy docs.
