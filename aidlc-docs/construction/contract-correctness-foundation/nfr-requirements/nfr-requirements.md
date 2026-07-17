# Unit 1 NFR Requirements

- **Determinism**: target alignment, utility order, and weighted results are
  stable across runs and supported Python versions.
- **Performance**: aggregation remains linear in period count, target rows, and
  utility rows; no solver or I/O is introduced.
- **Reliability**: structural mismatches fail before partial output is returned;
  optional diagnostics degrade to `None`.
- **Maintainability**: field policies are centralized and named; drift tests
  explain unclassified public additions.
- **Usability**: errors use public field/method vocabulary and never expose an
  internal enum as the remedy.
- **Availability/scalability**: N/A for an in-process local library operation.
- **Security/compliance**: Security extension disabled; aggregation processes
  already validated in-memory models and performs no external access.
