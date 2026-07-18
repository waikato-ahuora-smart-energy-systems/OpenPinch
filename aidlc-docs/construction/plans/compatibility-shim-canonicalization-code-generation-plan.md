# Compatibility Shim Canonicalization Code Generation Plan

## Unit Context

This single brownfield unit depends on the completed package-usability facade and
touches domain, contracts, application, analysis, presentation, tutorials, RTD, and
tests. It owns no database or infrastructure. The user's implementation request and
prior blanket approval authorize this complete checklist.

## Execution Checklist

- [x] **Step 1: Stage artifacts and contract inventory** — record requirements,
  workflow, functional design, approval, exact removal inventory, and partial-PBT
  obligations.
- [x] **Step 2: Enum and Value canonicalization** — replace every production, test,
  script, and documentation alias with full enum identities; remove `Value.values` and
  migrate consumers.
- [x] **Step 3: Descriptive Stream runtime** — replace public constructor arguments,
  properties, setters, helpers, and all callers while retaining private storage and
  compact wire fields.
- [x] **Step 4: Strict transport and workspace contracts** — forbid unknown fields,
  require schema version `3`, and add exact failure and round-trip tests.
- [x] **Step 5: Summary, graph, and design-view cleanup** — remove format strings,
  graph/location aliases, dynamic result forwarding, and view serialization.
- [x] **Step 6: Tutorials and documentation** — regenerate canonical notebooks and
  coverage data, remove seven legacy pages, and update RTD, release notes, and
  conventions.
- [x] **Step 7: Closed-contract and property tests** — add stale-symbol, reflection,
  strict-input, serialization, ordering, and mutation properties using seed
  `20260715`.
- [x] **Step 8: Focused verification** — run domain, contracts, application, graph,
  reporting, HPR, HEN, notebooks, documentation, Ruff, and patch-hygiene checks.
- [x] **Step 9: Complete build and test** — run the non-solver suite, applicable
  opt-in HPR/HEN profiles, warning-free Sphinx, distributions, and isolated wheel
  smoke.
- [x] **Step 10: Evidence closure** — complete implementation and build/test
  summaries, state, audit, extension compliance, and every remaining checkbox.

## PBT Compliance

- **PBT-02**: compact input and HEN serialization round trips.
- **PBT-03**: stream mutation, period ordering, and closed-inventory invariants.
- **PBT-07**: centralized domain-specific strategies.
- **PBT-08**: shrinking enabled and fixed seed `20260715`.
- **PBT-09**: existing Hypothesis/pytest integration.
