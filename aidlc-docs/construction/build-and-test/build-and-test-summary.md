# Build and Test Summary

## Build Status

- Build tool: uv and Hatchling.
- Runtime: CPython 3.14.2 for the complete repository suite; CPython 3.14.3
  for isolated artifact verification.
- Project version: OpenPinch 0.6.4.
- Build status: success.
- Source artifact:
  `/private/tmp/openpinch-step17.cUgtgY/artifacts/openpinch-0.6.4.tar.gz`.
- Wheel artifact:
  `/private/tmp/openpinch-step17.cUgtgY/artifacts/openpinch-0.6.4-py3-none-any.whl`.
- Source SHA-256:
  `78ff433120b332acc788b70916d879ed6da463f988266cf4b06b598a03bbd040`.
- Wheel SHA-256:
  `03f0557f2df029862a7bf08fa9fe3daee9abb70c224ac4abf17849b644a85069`.
- Archive verification: zero duplicate members; versioned schemas, fixtures,
  target modules, map modules, and compressor characteristic are present and
  byte-identical to the checkout.

## Test Execution Summary

### Complete Repository Suite

- Collected: 3,063 tests.
- Passed: 3,059.
- Skipped: 4 expected optional/environment profile tests.
- Failed: 0.
- Hypothesis seed: `20260715`, with normal shrinking enabled.
- Solver coverage: configured IPOPT, CBC, Bonmin, Couenne, APOPT, MojoPSE, and
  AMPL function libraries were available to solver-marked tests.
- Duration: 458.40 seconds.
- Status: pass.

### HPR Unit, Property, and Integration Coverage

- The focused heat-pump, application, contract, architecture, packaging, and
  resource selection passes 871 tests.
- Combined statement and branch coverage over the new Unit 3 modules and
  materially changed targeting paths is 97 percent, above the required
  95-percent minimum.
- The full run passes CoolProp default/explicit equivalence, pure and registered
  blend handling, explicit binary/ternary/N-component molar mixture contracts,
  TESPy target and map integration, exact cache behavior, lifecycle cleanup,
  target records, basis conversion, public accessor, multi-period boundaries,
  reporting, and plain JSON round trips.
- Status: pass.

### Integration Correction

The first complete run produced 3,058 passes and one tutorial-manifest failure.
The new specialist `hpr_performance_map` method was correctly in the live API
inventory but intentionally absent from notebook 09 because it requires a
specialist request contract that is not exported from the package root.

The manifest now distinguishes 196 notebook-demonstrated operations from this
documented and pytest-executable specialist API. The focused correction passes
6 tests, and the subsequent complete 3,063-test run passes. No notebook or root
export was added.

### Performance and Resource Stability

- A 10,000-point fake map passes its 5-second and 256-MiB limits.
- The exact call-local LRU remains bounded at 512 values and below 64 MiB traced
  Python memory for maximum-size fake results.
- Candidate callbacks and unique solve counts satisfy the linear bounds.
- Ten fake calls and at least three guarded real calls release private engine
  state after cleanup.
- The dedicated real public TESPy target plus two-point map passes under a
  Python-enforced 300-second timeout in 1.99 seconds.
- Status: pass.

### Documentation and Static Quality

- Sphinx 9.1.0 builds all 55 documentation sources with warnings treated as
  errors.
- Ruff lint passes for the complete repository.
- The changed Unit 3 Python surface is fully Ruff-formatted. A repository-wide
  format audit identifies 14 unrelated pre-existing files outside this unit.
- Python compilation, workflow YAML parsing, architecture dependency rules,
  package resource checks, and `git diff --check` pass.
- Status: pass.

### Installed Artifact Profiles

- Core profile: an isolated 18-package installation imports OpenPinch from
  site-packages, excludes TESPy, and passes root/default targeting, schema,
  fixture, plain-map, resource, problem, workspace, and CLI checks.
- TESPy profile: an isolated 33-package installation with TESPy 0.11.2 imports
  from site-packages and passes the explicit public target, winning record,
  target-derived basis, and two-point performance-map workflow.
- Checkout, core wheel, and TESPy wheel expose identical target/map signatures
  and resource digests.
- Status: pass.

## Generated Instructions

- `aidlc-docs/construction/build-and-test/build-instructions.md`
- `aidlc-docs/construction/build-and-test/unit-test-instructions.md`
- `aidlc-docs/construction/build-and-test/integration-test-instructions.md`
- `aidlc-docs/construction/build-and-test/performance-test-instructions.md`
- `aidlc-docs/construction/build-and-test/build-and-test-plan.md`

## Delivered Boundary

OpenPinch owns pinch targeting, optional CoolProp/TESPy thermodynamic
evaluation, detached winning records, and versioned plain-data performance
maps. CoolProp remains the default; TESPy is explicit and optional. OpenUtility
consumes exported mapping/JSON values and independently owns multi-period
dispatch, electricity and thermal balances, piecewise-linear MILP formulation,
Pyomo models, and HiGHS solves. OpenPinch imports none of those downstream
dependencies.

## Additional Tests

- Contract tests: pass.
- End-to-end public workflow tests: pass.
- Installed source/wheel tests: pass.
- Security tests: N/A; Security extension disabled and no new security boundary.
- Network load/stress tests: N/A; OpenPinch is a local library.

## Extension Compliance

- Property-Based Testing: compliant. Applicable PBT-01 through PBT-10 example,
  generated, stateful, oracle, replay, ordering, isolation, and performance
  requirements pass.
- Security: disabled; N/A.
- Resiliency: disabled; N/A.
- Blocking enabled-extension findings: none.

## Overall Status

- Build: success.
- All required tests: pass.
- Documentation: pass.
- Distribution verification: pass.
- Ready for Operations placeholder review: yes.
