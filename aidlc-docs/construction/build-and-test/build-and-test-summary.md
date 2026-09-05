# Build and Test Summary

## Build Status

- Build tool: uv with Hatchling through PEP 517.
- Runtime: CPython 3.14.2.
- Project version: OpenPinch 0.6.4.
- Build status: success.
- Artifact directory: `/private/tmp/openpinch-rtd-build.G9fYpj`.
- Source artifact: `openpinch-0.6.4.tar.gz`, 500471 bytes, SHA-256
  `3ebb52c1bec2db0de848ec78d5713d6243619e889c21e16c1965650069f8a609`.
- Wheel artifact: `openpinch-0.6.4-py3-none-any.whl`, 696693 bytes,
  SHA-256
  `6ff2b28c6afca26513997761ffcf59db5ebad59defe0af7cd2d808ed092482f5`.
- Archive verification: members are unique and notebook 09 is present in both
  artifacts. RTD source is repository-owned and verified separately by Sphinx.

## Test Execution Summary

### Complete Repository Suite

- Collected: 3064 tests.
- Passed: 3060.
- Skipped: 4 expected optional/environment profile tests.
- Failed: 0.
- Hypothesis seed: `20260715`, with normal shrinking enabled.
- Solver coverage: the configured external-solver environment was supplied to
  solver-marked tests.
- Duration: 448.34 seconds.
- Status: pass.

### Comprehensive Notebook 09

- All five code cells compiled and executed in order from a clean temporary
  directory.
- Runtime: 27.099 seconds, below the 300-second tutorial budget.
- The packaged sample produced guarded numerical infeasibility for its CoolProp,
  TESPy, refrigeration, and Brayton screens. The notebook completed without an
  uncaught exception, backend fallback, or partial performance map.
- The pure-fluid target, provider-registered blend syntax, explicit binary
  molar mixture, winning-record branch, three-point request, plain JSON export,
  and OpenUtility boundary remain present in executable source.
- Status: pass.

### Real TESPy Target-to-Map Oracle

- The dedicated public smoke created a successful explicit TESPy target,
  retained its detached winning record, and generated two ordered part-load map
  points.
- Result: 1 passed in 1.96 seconds.
- Enforced budget: less than 300 seconds.
- Status: pass.

### Documentation and Static Quality

- Sphinx 9.1.0 built all 55 RTD sources with warnings treated as errors.
- The complete repository Ruff lint gate passed.
- All four changed Python/test files pass Ruff format checks.
- Changed Python/test files compile.
- Generator repeated output is byte-idempotent.
- `git diff --check` passes.
- Status: pass.

## Test Categories

- Unit tests: pass through the complete repository suite.
- Property-based tests: pass with seed `20260715`.
- Integration and contract tests: pass, including target-record-map,
  public-import, manifest, resource, and documentation boundaries.
- End-to-end tests: pass, including clean notebook execution and the real TESPy
  public smoke.
- Performance tests: pass for the 27.099-second notebook execution and
  1.96-second real target-to-map smoke.
- Security tests: N/A because the Security extension is disabled and the change
  adds no security boundary.
- Network load/stress tests: N/A because OpenPinch is a local Python library.

## Generated Instructions

- `aidlc-docs/construction/build-and-test/build-instructions.md`
- `aidlc-docs/construction/build-and-test/unit-test-instructions.md`
- `aidlc-docs/construction/build-and-test/integration-test-instructions.md`
- `aidlc-docs/construction/build-and-test/performance-test-instructions.md`
- `aidlc-docs/construction/build-and-test/build-and-test-plan.md`

## Delivered Boundary

OpenPinch owns pinch targeting, the CoolProp default, explicit optional TESPy
thermodynamic evaluation, detached winning records, and versioned plain-data
performance maps. OpenUtility consumes the exported mapping/JSON structure and
independently owns multi-period dispatch, electricity and thermal balances,
piecewise-linear MILP formulation, Pyomo models, and HiGHS solves. Neither
package needs to import the other.

## Extension Compliance

- Property-Based Testing: compliant. Applicable generated, invariant,
  round-trip, ordering, lifecycle, cache, and oracle tests pass with the fixed
  seed.
- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Blocking enabled-extension findings: none.

## Overall Status

- Build: success.
- All required tests: pass.
- Documentation: pass.
- Distribution verification: pass.
- Ready for Operations placeholder review: yes.

This summary contains no Mermaid, ASCII diagram, or embedded JSON/YAML block.
Markdown headings, lists, paths, identifiers, hashes, and special characters
were validated before creation.
