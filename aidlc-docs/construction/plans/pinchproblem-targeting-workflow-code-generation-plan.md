# Unit 2 Code Generation Plan

- [x] **Step 1: Public contract tests** — extend the closed target vocabulary
  test to the live accessor, forbidden names, signatures, and root-only examples.
- [x] **Step 2: Effective argument foundation** — add the omitted sentinel,
  precedence/provenance resolver, read-only configuration view, and generated
  precedence properties.
- [x] **Step 3: Explicit heat-integration surface** — remove callable target,
  implement focused direct/indirect/Total Site and dependency-aware all-heat
  methods, and remove default/configured analysis selection.
- [x] **Step 4: Model-specific HPR surface** — implement Carnot,
  vapour-compression, Brayton, MVR, and symmetric refrigeration callables with
  named load validation and ephemeral backend option encoding.
- [x] **Step 5: Explicit advanced targeting** — rename area/cost and add
  specialized cogeneration methods plus object-oriented exergy/energy-transfer
  inputs.
- [x] **Step 6: All-period mirror** — replace top-level string dispatch with
  `target.all_periods`, ordered cached period results, deterministic workers,
  and selected-state restoration.
- [x] **Step 7: Remove targeting selectors** — delete the targeting-plan
  registry and targeting-enabled configuration fields, reject retired keys, and
  migrate samples and tests.
- [x] **Step 8: Lifecycle and observation** — simplify canonical serialization,
  configuration updates, invalidation, and actionable no-result guards without
  implicit solve.
- [x] **Step 9: Focused verification** — run target, period, configuration,
  HPR, reporting, contract, generated-property, Ruff, architecture, and stale-
  symbol gates with seed `20260715`.
- [x] **Step 10: Generation summary** — record changed files, parity evidence,
  intentional breaks, and extension compliance.

## PBT Generation Requirements

- PBT-02: round-trip effective run metadata and period results where serialized.
- PBT-03: precedence, mutual exclusion, ordering, determinism, and non-mutation.
- PBT-07: centralized public-argument and period-run strategies.
- PBT-08: normal shrinking and fixed verification seed.
- PBT-09: existing Hypothesis and pytest integration.

The user's blanket approval authorizes this full plan through completion.
