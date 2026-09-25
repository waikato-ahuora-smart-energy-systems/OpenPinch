# Test-Suite Runtime Reduction NFR Design Plan

- [x] Map the ordinary and specialized test selections to single CI owners.
- [x] Design work-elimination boundaries for Notebook 19 and repeated optimizer
  searches.
- [x] Design copy-on-consume fixture ownership and mutation-isolation checks.
- [x] Design reduced-budget acceptance around feasibility and convergence
  invariants rather than brittle coordinates.
- [x] Design batched child-process execution with per-case diagnostics.
- [x] Design coverage protection and the conditional fallback path.
- [x] Design explicit timeout and failure behavior without retries or hidden
  skips.
- [x] Map logical components, dependency order, and verification ownership.
- [x] Evaluate resilience, scalability, performance, security, infrastructure,
  and Property-Based Testing compliance.

No clarification questions are required:

- **Resilience**: approved requirements mandate visible failure, no retry, and
  finite timeout behavior.
- **Scalability**: the bounded serial target and linear logical-case growth are
  explicit.
- **Performance**: the overall and focused same-host budgets are quantitative.
- **Security**: no security surface changes and the Security extension is
  explicitly disabled.
- **Logical components**: the existing generator, tests, pytest configuration,
  and three workflow files are named owners; no infrastructure component is
  introduced.
