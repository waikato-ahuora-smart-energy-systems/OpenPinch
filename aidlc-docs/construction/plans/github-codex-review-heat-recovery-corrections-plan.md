# GitHub Codex Review Heat-Recovery Corrections Plan

## Objective

Resolve both unresolved Codex findings on pull request 95 without changing the
public API or disturbing the concurrent pull-request version-bump work.

## TDD Execution

- [x] Retrieve the current review and validate each finding against the code.
- [x] Add a failing near-limit regression proving the inverse solver retains
  the user's requested recovery as its feasibility target.
- [x] Correct near-limit inversion while retaining its documented status.
- [x] Add failing output-unit regressions for Pint-compatible Celsius and
  Fahrenheit aliases.
- [x] Resolve configured temperature units through Pint and convert absolute
  aliases to their corresponding delta units.
- [x] Run focused tests, formatting, lint, and the applicable full test gate.
- [x] Record final implementation and verification evidence in AI-DLC state and
  audit artifacts.

## Extension Assessment

- Property-Based Testing: enabled; existing heat-recovery properties remain
  applicable, while deterministic regressions are the strongest oracle for
  these two concrete review findings.
- Security Baseline: disabled by existing extension configuration.
- Resiliency Baseline: disabled by existing extension configuration.
