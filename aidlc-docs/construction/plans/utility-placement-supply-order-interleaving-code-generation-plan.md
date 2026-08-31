# Utility Placement Supply-Order Interleaving Code Generation Plan

- [x] Step 1: Record the supply-temperature ordering amendment and add a
  notebook-derived failing regression proving that a lower-temperature
  isothermal level may interleave with a sensible level.
- [x] Step 2: Remove declaration-order coupling between generated utility
  kinds, retain stable same-kind identities and separation, and construct
  deterministic interleaved starts; place residual `HU`/`CU` supplies 50 K
  beyond the context-wide process-temperature extremes.
- [x] Step 3: Run focused property, codec, optimisation, and application tests;
  reproduce the modified notebook case without rewriting the user's notebook.
- [x] Step 4: Run broad regression and quality gates, review the final diff for
  necessary changes only, update evidence and state, and commit to `develop`.
- [x] Step 5: Reproduce the saved notebook failure and add RED tests requiring
  candidate-specific balanced-composite failures to be scored as infeasible
  without hiding invariant thermodynamic defects.
- [x] Step 6: Contain recoverable candidate thermodynamic failures at the
  evaluation boundary with typed diagnostics, then rerun the exact uncapped
  Process and Site notebook workflows without rewriting the user's notebook.
- [x] Step 7: Run focused and broad regression gates, update only the necessary
  requirements, design, state, audit, and build evidence, review the diff, and
  commit the correction to `develop`.
- [x] Step 8: Trace the unbalanced candidate to its first non-conserving
  operation and add a RED regression using a near-isothermal utility endpoint
  adjacent to a process-table breakpoint.
- [x] Step 9: Reconstruct utility composite profiles from exact allocated duty
  and temperature fractions, independent of problem-table interval filtering,
  and prove endpoint duty conservation with deterministic and property tests.
- [x] Step 10: Rerun the exact uncapped notebook workflows and broad gates,
  replace containment-only evidence with the root-cause result, review only
  necessary changes, and commit the correction to `develop`.
