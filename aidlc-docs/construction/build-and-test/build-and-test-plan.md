# Repository Issue Remediation Build and Test Plan

## Applicability Assessment

- Unit tests: applicable to application, workspace, reporting, import isolation,
  and documentation guards.
- Integration/contract tests: applicable to root API, architecture, HEN
  transport, workspace bundles, reporting, packaging, and tutorials.
- End-to-end tests: included in the complete non-solver suite and installed
  artifact smoke.
- Performance load/stress tests: N/A; no service, latency objective, throughput
  requirement, or algorithmic complexity change.
- Security tests: N/A because the Security Baseline extension is disabled; the
  concrete input/path protections are covered as functional contracts.
- External solver execution: unchanged and outside this remediation's required
  acceptance gate. Solver-facing contracts remain included in non-solver tests.

## Execution Checklist

- [x] Analyze unit, integration, contract, end-to-end, performance, security,
  documentation, and distribution requirements.
- [x] Generate build instructions.
- [x] Generate unit/property test instructions.
- [x] Generate integration/contract/end-to-end instructions.
- [x] Record performance and external-solver applicability.
- [x] Run the complete fixed-seed non-solver suite.

**Test evidence**: the final post-correction run passed 2,181 tests, with 3
explicitly guarded optional-profile tests skipped and 4 solver-marked tests
deselected in 148.03 seconds using Hypothesis seed `20260715`.
- [x] Run repository Ruff lint and format checks.
- [x] Run patch hygiene and stale current-contract scans.

**Static evidence**: Ruff lint passed, all 460 Python files were already
formatted, `git diff --check` reported no findings, and the closed current-
contract scan returned no matches.
- [x] Build warning-free Sphinx HTML from a clean temporary destination.

**Documentation evidence**: Sphinx 9.1.0 read and built all 53 sources with
`--fail-on-warning`; the clean HTML output completed with zero warnings.
- [x] Build wheel and source distribution into a clean temporary destination.
- [x] Install and smoke the wheel outside the checkout.
- [x] Generate the final build-and-test summary.
- [x] Update state, audit, requirements traceability, and Operations N/A.

**Artifact evidence**: `openpinch-0.5.2-py3-none-any.whl` and
`openpinch-0.5.2.tar.gz` built successfully in a clean temporary directory. The
wheel was installed with its declared runtime dependencies into a temporary
Python 3.14 environment and passed the workflow, resources, two-name root API,
retired-package absence, and CLI smoke outside the checkout.

**Focused HEN evidence**: 458 non-external OpenHENS/HEN tests passed with four
solver-marked deselections in 15.86 seconds.

## Post-Gate Contract Audit

- [x] Reproduce generic-mapping bundle validation bypass.
- [x] Validate all mapping-shaped bundle inputs and nested case mappings.
- [x] Rerun focused workspace, static, complete non-solver, distribution, and
  installed-wheel gates.

**Discovery**: a final manual probe showed that Pydantic accepts generic
`Mapping` inputs while the pre-validator inspected only concrete `dict`
instances. A read-only mapping containing `../escape` was therefore accepted.
The completed result is reopened until this edge case is closed.

**Closure evidence**: the regression failed before the correction, then 158
focused workspace/contracts tests passed. The complete suite passed 2,181 tests,
the corrected artifacts rebuilt, and the installed wheel passed both the
standard smoke and an explicit generic-mapping rejection probe.

The user's `Continue to Completion` authorization covers these approved gates
and the final Operations N/A transition.
