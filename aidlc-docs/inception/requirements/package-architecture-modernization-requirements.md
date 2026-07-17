# Package Architecture Modernization Requirements

## Status

Approved by the user's explicit `Go.` response to the detailed Code Generation
checklist. All ten implementation steps and the final quality, test, build, and
distribution gates were completed on 2026-07-17. Generated code is awaiting the
workflow's explicit user review.

## Functional Requirements

- **ARCH-01 External contract**: Keep
  `OpenPinch.main.pinch_analysis_service` as the sole compatibility-protected
  Python import. Keep `OpenPinch/main.py` minimal and preserve its signature,
  validation ordering, exceptions, output ordering, serialization, and
  numerical behaviour.
- **ARCH-02 Concrete ownership**: Place business state in `domain`, wire models
  in `contracts`, use-case coordination in `application`, engineering
  calculations in `analysis`, infrastructure translation in `adapters`, and
  rendering/reporting in `presentation`.
- **ARCH-03 Reusable optimisation**: Own scalar optimisation models, candidate
  handling, execution, and backends in `optimisation`. Heat-pump analysis must
  cross one explicit adapter; a non-HPR consumer must be able to reuse the
  optimiser without importing heat-pump code.
- **ARCH-04 No compatibility machinery**: Remove retired package trees,
  forwarding modules, dynamic export barrels, import aliases, and pickle-path
  shims. The package root and package initializers are import-free markers.
- **ARCH-05 Parent-owned records**: Keep stream segments, exchanger period
  states, exchanger area slices, Process MVR state, multiperiod state, graph
  build state, dashboard state, and solver state private to their owners.
- **ARCH-06 Composition over mixins**: Extract cohesive functions and state
  with explicit inputs. Do not add inheritance mixins, service locators,
  mutable registries, or speculative protocols.
- **ARCH-07 Behaviour preservation**: Preserve solver axes, equation and
  parameter order, warm starts, tolerances, period ordering, numerical
  fixtures, and result structures unless the clean-break contract explicitly
  removes an import path.
- **ARCH-08 Observable tests**: Make the main end-to-end suite authoritative.
  Organize tests by observable owner, retain focused mathematical-kernel tests,
  and reject tests that only pin a private helper path or forwarding call.
- **ARCH-09 Distribution**: Build wheel and sdist with all intended owners and
  packaged assets, no retired package, and no repository-only application.
- **ARCH-10 Documentation**: Document the sole protected contract, dependency
  directions, unsupported advanced owners, and the 0.5.0 no-migration policy.

## Quality Requirements

- Score Ease of Change, Behavioural Tests, Clear Boundaries, Low Coupling, and
  Project Coherence at least 9/10; score Simplicity at least 8/10; score overall
  quality at least 8.8/10.
- Pass AST dependency-direction, exact-exception, cold-import, retired-path,
  and no-facade gates.
- Pass the complete non-solver suite at no less than 95 percent statement
  coverage without branch-coverage regression.
- Keep partial PBT rules PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 blocking,
  with Hypothesis seed `20260715` and shrinking enabled.
- Pass available solver, canonical HPR/HEN, Ruff, warning-free Sphinx,
  notebook-parse, isolated-build, clean-wheel-install, and patch-hygiene gates.

## Extension Configuration

- Security Baseline: disabled; N/A for this implementation.
- Resiliency Baseline: disabled; N/A for this implementation.
- Property-Based Testing: partially enabled and blocking for PBT-02, PBT-03,
  PBT-07, PBT-08, and PBT-09.

## Verification Traceability

| Requirement | Verification |
|---|---|
| ARCH-01 | The authoritative 59-case source and clean-wheel suites import only `OpenPinch.main.pinch_analysis_service`; signature, validation, serialization, ordering, and representative numerical values pass. |
| ARCH-02 | AST layer-direction tests and concrete-owner tests pass for `domain`, `contracts`, `application`, `analysis`, `adapters`, and `presentation`. |
| ARCH-03 | Generic convex and generated scalar objectives exercise `optimisation` without heat-pump imports; HPR uses one explicit optimisation adapter. |
| ARCH-04 | Root and owner initializers are import-free markers; retired modules, aliases, dynamic barrels, and pickle shims are absent from source and artifacts. |
| ARCH-05 | Parent-owned record tests pass for stream segments, exchanger period states, exchanger area slices, Process MVR state, graph state, and solver state. |
| ARCH-06 | Composition helpers receive explicit inputs; architecture tests reject parent-barrel imports and cross-owner concrete re-exports. |
| ARCH-07 | Exact HPR/HEN fixtures, solver construction-order checks, and the available solver suite pass. |
| ARCH-08 | Tests mirror observable owners; the main end-to-end suite is the only compatibility suite and private tests are limited to mathematical or solver semantics. |
| ARCH-09 | Isolated wheel and sdist each contain 326 intended entries and no retired package; the installed-wheel contract suite passes outside the checkout. |
| ARCH-10 | Architecture, support policy, release notes, notebooks, examples, and advanced API guidance document the 0.5.0 clean break. |
