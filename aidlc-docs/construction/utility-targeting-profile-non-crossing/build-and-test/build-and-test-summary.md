# Utility Targeting Profile Non-Crossing Build and Test Summary

## Build status

- Locked uv environment: success.
- Ruff: success.
- Patch whitespace validation: success.
- Warning-clean documentation build: success through the full packaging suite.
- Notebook generator repeatability and executable notebook 19: success.
- Release artifact and package-resource checks: success through the complete
  repository suite.

## Test execution

- Shared targeting and direct targeting: 67 passed.
- Utility-placement analytical and shared targeting gate: 248 passed.
- Selected application exact-replay workflows: 7 passed in 91.63 seconds.
- Notebook and documentation consistency gate: 20 passed in 30.32 seconds.
- Complete solver-enabled suite: 2,450 passed, 4 expected skips, 0 failures in
  526.67 seconds.

## Correctness evidence

- The failing 174.01 to 62.9556 degC sensible profile is now limited by the
  interior Process GCC point near 97 degC.
- Generated hot and cold examples prove the Utility GCC does not exceed the
  corresponding residual profile at any breakpoint.
- Public Process exact replay checks `H_NET_UT <= H_NET_A` within the existing
  numerical tolerance.
- Existing maximum-duty, fallback, direct-targeting fixture, Process, Site,
  Community, Region, multiperiod, and batch behavior remains green.
- Canonical notebook 19 is executed, assertion-free, generator-equivalent, and
  displays only the standard Process GCC and Total Site Profile plots.

## Overall status

- Build: success.
- Tests: pass.
- PBT compliance: pass with no blocking findings.
- Operations: N/A; no deployment or publication was requested.
