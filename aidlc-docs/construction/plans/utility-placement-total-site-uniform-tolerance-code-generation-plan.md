# Utility Placement Total Site Uniform-Tolerance Alignment Code Generation Plan

This plan is the single source of truth for the saved Total Site solve failure.
The correction is bounded to problem-table alignment and the existing utility-
placement replay path. The user's standing approval authorizes execution unless
an unexpected scope change occurs.

- [x] **Step 1 - Reproduce and isolate.** Read the saved notebook traceback,
  reproduce successful 50- and 100-generation fixed-seed runs, and shrink the
  late-convergence 70/71-row failure to one three-row target and four-row source
  containing distinct temperatures separated by 0.6 microkelvin.
- [x] **Step 2 - RED uniform-alignment coverage.** Add the shrunk example and a
  Hypothesis property requiring tolerance-equivalent grids, finite cumulative
  columns, endpoint conservation, and no broadcast mismatch for valid near-
  isothermal temperature clusters.
- [x] **Step 3 - GREEN uniform temperature tolerance.** Apply the canonical
  domain tolerance consistently to grid construction, near-duplicate removal,
  missing-interval detection, grid comparison, interpolation safeguards, and
  Total Site column alignment. Do not introduce a private zero-tolerance path.
- [x] **Step 4 - Total Site workflow regression.** Run the fixed-seed four-
  isothermal Total Site utility-placement workflow through the generation count
  that exposed the saved failure and verify it returns a feasible detached case.
- [ ] **Step 5 - Build, review, and completion.** Run focused targeting and
  utility-placement tests, fixed-seed properties, Ruff, patch hygiene, the
  applicable broad suite, documentation validation, review scope, preserve the
  user's notebook and `.gitignore`, update records, and commit to `develop`.

## PBT Compliance Plan

- PBT-01: canonical-grid size/order and cumulative-profile conservation are the
  identified invariants.
- PBT-03: equivalent inputs must produce consistently deduplicated, descending,
  finite grids and aligned cumulative columns.
- PBT-05: canonical rounded clustering under the shared domain tolerance is the
  reference oracle.
- PBT-07: generated grids contain valid descending endpoints and constrained
  sub-tolerance near-isothermal clusters.
- PBT-08: Hypothesis shrinking remains enabled and the repository fixed seed is
  used for verification.
- PBT-09: use the existing Hypothesis and pytest stack.
- PBT-10: retain the shrunk saved failure as an explicit regression alongside
  the generated property.
- PBT-02, PBT-04, and PBT-06: N/A; no inverse, idempotent public operation, or
  stateful command model is introduced.
