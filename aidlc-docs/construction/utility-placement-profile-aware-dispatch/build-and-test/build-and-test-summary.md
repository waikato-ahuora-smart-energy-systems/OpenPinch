# Utility Placement Profile-Aware Dispatch Build and Test Summary

## Build status

- Status: success
- Tool: `uv build`
- Artifacts: OpenPinch 0.5.4 source archive and wheel
- Archive check: notebook 19 is present in both distributions
- RTD: 54-page Sphinx HTML build passed with warnings treated as errors

## Test results

- Focused utility-placement gate: 220 passed in 56.07 seconds
- Notebook, documentation, and graph regressions: 41 passed, 3 guarded skips
- Performance gate: passed, including the 250 ms p95 pure-model requirement
- Ruff: check and format-check passed on all intentionally changed Python files
- Patch hygiene: `git diff --check` passed
- Complete solver-enabled suite: 2,445 passed, 4 expected skips, 0 failures in
  306.33 seconds

## Notebook evidence

Notebook 19 executed from top to bottom with the repository Python 3.14 kernel.
The capped Process result has fallback penalty below 0.5 and uses at least
three feasible named hot levels under standard targeting. The Total Site
placement objective is below 0.65 kW/K and includes a cold sensible span above
5 degC before standard Total Site targeting and plotting.

## Overall status

Build and test passed. No external services, infrastructure, or deployment work
is required. Security and Resiliency extensions are disabled; enabled
Property-Based Testing checks are compliant.
