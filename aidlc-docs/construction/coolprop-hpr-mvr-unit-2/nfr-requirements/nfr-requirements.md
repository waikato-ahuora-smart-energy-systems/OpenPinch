# Unit 2 NFR Requirements

| ID | Requirement |
|---|---|
| U2-NFR-01 | Default limits remain 300 iterations and 1,000,000 evaluations for compatibility. |
| U2-NFR-02 | Explicit limits are validated before any optimiser or property call. |
| U2-NFR-03 | Invalid fluid/state preflight performs zero generic-optimiser calls. |
| U2-NFR-04 | Exact repeated coordinates perform at most one search objective evaluation per request. |
| U2-NFR-05 | Search cache and diagnostics are request-local and released after completion. |
| U2-NFR-06 | Representative diagnostics are capped at 16 independent of evaluation count. |
| U2-NFR-07 | Public failures contain no traceback, engine, or arbitrary exception object. |
| U2-NFR-08 | Unexpected internal exceptions preserve their Python type and cause. |
| U2-NFR-09 | A viable warm start remains available after backend no-candidate/budget exhaustion. |
| U2-NFR-10 | Existing calls without new arguments remain source-compatible. |
| U2-NFR-11 | `HPRTargetingError` remains catchable as `ValueError`. |
| U2-NFR-12 | Single- and multiperiod paths resolve the same defaults and validation. |
| U2-NFR-13 | TESPy behavior and selection remain unchanged. |
| U2-NFR-14 | Focused deterministic tests must not depend on wall-clock timing. |
| U2-NFR-15 | Real-engine timing is documented separately from deterministic call-count gates. |
| U2-NFR-16 | Changed production code passes Ruff, formatting, compilation, and focused regression tests. |
| U2-NFR-17 | Fixed-seed properties cover ordering, caching, failure sequence, caps, and precedence. |
| U2-NFR-18 | No new package, service, process, or database is introduced. |
