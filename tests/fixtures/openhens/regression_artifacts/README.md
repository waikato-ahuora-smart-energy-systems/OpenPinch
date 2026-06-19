OpenHENS Regression Artifacts
=============================

This directory contains checked-in evidence used by the OpenHENS migration
regression tests. These files are fixtures, not generated output from normal
test runs.

Directory roles:

* ``openhens_solver_runs/`` stores compact metrics copied from source OpenHENS
  solver runs. HENS-11 tests use these files to preserve the expected objective,
  dTmin, stage, unit-count, and solved-case contracts.
* ``adapter_array_snapshots/`` stores source-array parity snapshots for the
  private solver array adapter. Adapter tests use these files to verify that
  converted OpenPinch fixtures prepare the same solver arrays by stream and
  utility identity.
* ``network_design_snapshots/`` stores expected OpenPinch-native
  ``HeatExchangerNetwork`` snapshots for selected best ESM solutions. HENS-08
  and HENS-11 tests use these files to verify extracted topology, utility
  loads, areas, and objective values.
