Tutorial Coverage Map
=====================

Coverage Result
---------------

The released public denominator contains **211 operations** across
``PinchProblem``, ``PinchWorkspace``, targeting, all-period targeting,
components and returned component results, ordered case batches, HEN
design/result views, and plotting. The canonical manifest maps every operation
except those still under development to an executable tutorial code cell:
**199/211 operations are mapped**. Routine fake and guarded
real-engine pytest profiles provide
additional evidence for the specialist target-owned HPR map operation.

Operation coverage and notebook execution coverage are different. The ``base``
profile is executed in routine CI. ``slow-hpr``, ``solver``, and ``interactive``
have executable opt-in gates and require their declared environment. A profile
is not reported as executed merely because its operations are mapped. The
``execution_evidence`` column distinguishes routine execution, opt-in profiles,
and batch-delegation contracts.

The two Brayton callables, and their batch mirrors, are marked
``unmapped; under development``. They remain public but raise while the Brayton
solver is under development, so no tutorial demonstrates them.

Counting Rules
--------------

- The live public classes and accessors form the denominator.
- Only executable code-cell references count; Markdown mentions do not.
- Every manifest operation must exist live and every primary tutorial must be
  packaged.
- A removed operation leaves the denominator only after it is absent from the
  live facade and tutorials. No compatibility alias is counted.
- Properties and accessors use the ``cached observation`` semantic mode.
- Named analysis/design methods use ``explicit execution``. Export and
  dashboard calls use ``explicit side effect``.
- Constructors and returned result operations are included; coverage is not
  limited to methods reached by a shallow accessor scan.
- Source type, zone scope, configuration precedence, placement, period scope,
  aggregation, workspace selection, HEN method, and plot behavior are tracked
  as separate semantic dimensions.

Execution Profiles
------------------

Every packaged notebook is executed by ``test_notebook_executes`` in
``tests/packaging/test_notebooks.py``. Each notebook carries the
``tutorial_profile`` marker plus the marker of the environment it needs:

- ``base`` and ``interactive``: no extra marker; run in the unit lane.
- ``slow-hpr``: ``tespy``; run in the ``notebooks-hpr`` lane.
- ``solver``: ``solver``; run in the solver job.

Run one profile locally with its declared extras installed, for example:

.. code-block:: bash

   uv run pytest tests/packaging/test_notebooks.py -m "tespy and tutorial_profile"

Use ``-m "solver and tutorial_profile"`` for the HEN notebooks, or
``-k test_notebook_executes`` for the whole series.

Release Verification Snapshot
-----------------------------

For this release candidate, all declared profiles were executed from clean
temporary directories:

- ``base``: 10 notebooks, passed in the routine non-solver suite;
- ``slow-hpr``: 4 notebooks, passed;
- ``solver``: 3 HEN notebooks, passed with the synthesis environment; and
- ``interactive``: 1 notebook, passed with dashboard launch guarded while real
  plot and workbook exports executed.

Numerical infeasibility is a valid screening result and is retained with its
reason. It is distinct from an unsupported method and from a test failure.

Canonical Manifest
------------------

.. csv-table:: Public operation to tutorial coverage
   :file: ../_data/tutorial-coverage.csv
   :header-rows: 1
   :class: longtable

API owners are documented in :doc:`../api/pinchproblem` and
:doc:`../api/pinchworkspace`. Tutorial owners are described in
:doc:`notebook-series`.
