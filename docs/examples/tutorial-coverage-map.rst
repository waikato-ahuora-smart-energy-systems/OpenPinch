Tutorial Coverage Map
=====================

Coverage Result
---------------

The released public denominator contains **186 operations** across
``PinchProblem``, ``PinchWorkspace``, targeting, all-period targeting,
components and returned component results, ordered case batches, HEN
design/result views, and plotting. The canonical manifest maps all 186 live
operations to tutorial code cells: **186/186, or 100 percent mapping
coverage**.

Operation coverage and notebook execution coverage are different. The ``base``
profile is executed in routine CI. ``slow-hpr``, ``solver``, and ``interactive``
have executable opt-in gates and require their declared environment. A profile
is not reported as executed merely because its operations are mapped. The
``execution_evidence`` column distinguishes routine execution, opt-in profiles,
and batch-delegation contracts.

The two Brayton callables, and their batch mirrors, are mapped but explicitly
marked ``runtime unsupported`` while the underlying solver contract is under
repair. Tutorial 09 demonstrates handling that status as a screening outcome;
the map does not misreport those callables as successful analyses.

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

Run an optional profile only in an environment containing its declared extras:

.. code-block:: bash

   OPENPINCH_TUTORIAL_PROFILES=slow-hpr uv run pytest \
     tests/packaging/test_notebooks.py -k optional_profile

Use ``solver`` or ``interactive`` in place of ``slow-hpr``. The routine gate
always executes every ``base`` notebook from a clean temporary directory.

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
