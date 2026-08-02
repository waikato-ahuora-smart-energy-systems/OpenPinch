# Notebook Presentation Performance Test Instructions

## Applicability

No load, concurrency, service-throughput, or scalability requirement applies to
this local tutorial change. Performance validation is limited to ensuring that
result presentation does not rerun engineering analysis and that notebook
profiles remain within their declared qualitative runtime bands.

## Runtime Profile Checks

- Base notebooks: execute through the routine focused suite.
- Slow-HPR notebooks: record the selected profile duration.
- Solver notebooks: record the selected profile duration with provisioned
  external solvers.
- Interactive notebook: execute with the existing test-safe dashboard boundary.

## Pass Criteria

- Review cells display cached or already-computed values only.
- No review cell calls a `target` or `design` engineering method.
- No new optimization restart, stage, or search expansion is introduced.
- Profile runtimes remain consistent with declared notebook expectations.

## N/A Performance Measures

HTTP response time, requests per second, concurrent users, database throughput,
and service error rate are N/A because OpenPinch notebooks are local workflows,
not a hosted service.
