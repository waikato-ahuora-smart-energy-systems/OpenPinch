# Utility Placement Exact-Target Replay Build and Test Summary

## Build status

- Tool: `uv build`
- Status: passed
- Artifacts: `openpinch-0.5.4.tar.gz` and
  `openpinch-0.5.4-py3-none-any.whl`
- Distribution check: the wheel contains the executed assertion-free utility
  placement notebook.

## Verification results

- Ruff: passed with no findings.
- Patch hygiene: `git diff --check` passed.
- Sphinx: full clean HTML build passed with warnings treated as errors.
- Focused analytical and property suite: 181 passed.
- Notebook generator, execution metadata, documentation, manifest, and package
  gates: passed as part of the complete suite.
- Complete solver-enabled suite: 2,448 passed and 4 expected skips from 2,452
  collected tests in 612.17 seconds.
- Bounded exact Process replay measurement: 25.101 seconds for 115 accounted
  backend-plus-canonical evaluations with a 100-evaluation backend limit.

## Correctness evidence

- Process, Site, Community, and Region ordinary retargeting reproduce optimizer
  duties by utility name and fallback status.
- A capped two-period Process case reproduces the evidence in each period.
- Candidate-specific target totals and balanced-composite snapshots feed
  coverage, fallback penalty, entropy generation, and ranking.
- Requested levels may remain at zero duty; no minimum-duty API is introduced.
- Notebook 19 uses two isothermal and two non-isothermal sensible levels,
  displays standard GCC and Total Site Profile figures, and contains no tests or
  assertions.
- No monetary utility-placement optimization or utility-placement CLI was
  introduced.

## Extension compliance

- Property-Based Testing: compliant; focused properties and the complete suite
  pass with shrinking enabled.
- Security Baseline: N/A because it is disabled and no external boundary was
  added.
- Resiliency Baseline: N/A because it is disabled and no deployment or service
  boundary was added.

## Overall status

Build and all required quality gates pass. The correction is ready to commit to
`develop`.
