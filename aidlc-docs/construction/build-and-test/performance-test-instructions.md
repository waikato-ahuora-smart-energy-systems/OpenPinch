# Performance Test Instructions

## Applicability

No new server, concurrency service, latency service-level objective, or load
requirement was introduced. A standalone load or stress suite is therefore N/A
for this package usability refactor.

The relevant execution-cost checks remain functional:

- all-period methods preserve deterministic ordering for serial and threaded
  workers;
- notebook execution profiles separate base, optional, and solver-backed work;
- external solver tests remain marked and are not hidden inside the ordinary
  non-solver suite.

Any future performance baseline should record the exact case, operating-period
count, solver/backend, worker count, hardware, and dependency versions before
comparing elapsed time.
