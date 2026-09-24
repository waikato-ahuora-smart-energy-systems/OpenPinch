# Unit 1 Code Generation Summary

Unit: Candidate Correctness and Detached HPR Results

Status: Complete and approved under the user's authorization through completion.

## Outcome

The shared HPR boundary now preserves every finite penalty term, separates cheap
search evaluation from artifact-producing final evaluation, records generalized
CoolProp topology evidence, and prevents live thermodynamic engine objects from
crossing into caller-facing results.

## Implemented Changes

- Added strict reliability contracts for evaluation mode, topology, search
  budgets, classified diagnostics, targeting errors, and ordered loop/stage
  simulation evidence.
- Added deterministic scalar and rectangular penalty normalization and routed
  Carnot, simulated vapour-compression, and VC+MVR accounting through it.
- Propagated explicit `search` and `final` evaluation modes through single- and
  multiperiod adapters plus cascade, parallel, VC+MVR, and TESPy objectives.
- Suppressed model, figure, and simulation-record creation/retention during
  optimizer search while preserving numerical objective semantics.
- Generalized accepted CoolProp simulation records for cascade, parallel, and
  VC+MVR topologies with detached ordered loop/stage facts.
- Removed live model projection from public outputs, recursively finalized
  period and weighted results, and made deep-copy validation part of public
  translation.
- Routed the public HPR service through the shared detached finalizer.

## Traceability

| Obligation | Implementation evidence | Verification evidence |
|---|---|---|
| FR-1, FR-2 | shared penalty normalizer and accounting routes | rectangular examples, invalid-shape examples, Hypothesis invariant/idempotence/oracle properties |
| FR-3 | recursive output finalization and model omission | uncopyable-engine and nested-period regressions |
| FR-8 | public service uses shared detached translation | contract, adapter, multiperiod, and service-boundary regressions |
| FR-4, FR-5, FR-7 foundations | typed budgets, failure vocabulary, topology evidence | strict validation and JSON round-trip tests |
| NFR-U1-01 through NFR-U1-25 | pure helpers, explicit modes, frozen records, bounded values, no new dependencies | Ruff, formatting, compilation, patch hygiene, and focused regression gate |
| U1-P1 through U1-P12 | property implementations in reliability and contract tests | all categorized properties pass with fixed repository seeds |

## Verification

- Focused and compatibility tests: `197 passed`.
- Ruff: all selected changed production and test files pass.
- Ruff formatting: all selected files formatted.
- Python compilation: changed package and test modules compile.
- Patch hygiene: `git diff --check` passes.
- Dependencies: unchanged.
- Duplicate brownfield source files: none introduced.

## Extension Compliance

- Property-Based Testing: compliant. U1-P1 through U1-P12 are executable and
  retain deterministic seeds and shrinking.
- Security: disabled for this workflow; not applicable.
- Resiliency: disabled for this workflow; not applicable.

## Remaining Work

Unit 2 owns CoolProp state-capability preflight, bounded search budgets,
warm-start-first retention, failure aggregation, caching, and public optimized
service integration. Unit 3 owns direct process-MVR hardening and the integrated
real-engine and notebook proof.
