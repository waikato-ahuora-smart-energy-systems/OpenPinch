# Unit 3 Code Generation Plan

- [x] **Step 1: Component surface** — replace `add_component.process_mvr` with
  `components.add_process_mvr`, config fallback, inventory, and invalidation.
- [x] **Step 2: HEN design surface** — add descriptive method names, named
  design arguments, fixed prerequisites, and explicit multiperiod design.
- [x] **Step 3: Design result view** — add one-based `top`, `network`, selected
  metrics, and lazy `grid` behavior without changing result serialization.
- [x] **Step 4: Workspace case surface** — make scenarios unsolved, remove
  variant/workflow-string aliases, and align active-case forwarding.
- [x] **Step 5: Ordered case batches** — add `cases(names)` with mirrored
  target/design execution and deterministic per-case outcomes.
- [x] **Step 6: Observation and plot surface** — remove callable plot and
  `get_graph_data`, use `data`, integer indexes, named exports, and explicit
  period aggregation booleans.
- [x] **Step 7: Contract and generated tests** — update HEN/workspace/plot tests
  and add order, rank, non-mutation, and no-hidden-execution properties.
- [x] **Step 8: Focused verification** — run Unit 3 tests, Ruff, architecture,
  cold-import, stale-symbol, and patch-hygiene gates with seed `20260715`.
- [x] **Step 9: Generation summary** — record modified/created files,
  intentional breaks, parity evidence, and extension compliance.

## Context and Traceability

This brownfield unit implements US-3, US-4, US-5, and US-8 using existing
application, analysis, presentation, and test owners. It depends on completed
Units 1 and 2, owns no database or deployment entity, and preserves internal
numerical service boundaries. This checklist is the Code Generation source of
truth. The user's blanket approval authorizes every step through completion.
