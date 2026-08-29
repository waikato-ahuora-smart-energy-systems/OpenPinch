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
