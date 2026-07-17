# Business Rules

- The public component entry is `components.add_process_mvr`; mutation never
  targets and clears derived target, period, graph, report, and design state.
- Design algorithms are selected by descriptive methods, never configuration
  strings. Explicit kwargs override stored numerical assumptions ephemerally.
- Ranked design access is one-based; invalid ranks fail clearly and selection
  never mutates the synthesis result.
- Plot, summary, report, comparison, export, and dashboard calls never run a
  target or design method.
- Scenarios are unsolved. Batch execution preserves requested case order and
  records success or failure independently for each case.
- Case creation and inspection use case vocabulary only; variant/workflow-string
  APIs are not part of the process-engineer contract.
