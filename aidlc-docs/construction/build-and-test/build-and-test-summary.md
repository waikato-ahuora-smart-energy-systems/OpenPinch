# Build and Test Summary

## Build Status

- Build tool: Hatchling through uv and the repository build helper.
- Status: success.
- Artifacts: `openpinch-0.5.2-py3-none-any.whl` and
  `openpinch-0.5.2.tar.gz` from a clean temporary destination.
- Installed artifact: passed from a temporary Python 3.14 environment outside
  the source checkout.
- Documentation: Sphinx 9.1.0 built 53 sources with warnings treated as errors.

## Planned Gates

| Gate | Result | Status |
|---|---:|---|
| Complete fixed-seed non-solver suite | 2,181 passed, 3 skipped, 4 deselected | Pass |
| Repository Ruff lint/format | All checks; 460 files formatted | Pass |
| Patch and current-contract scans | No findings | Pass |
| Clean warning-as-error Sphinx build | 53 sources, zero warnings | Pass |
| Wheel and source distribution build | 2 artifacts, version 0.5.2 | Pass |
| Isolated installed-wheel smoke | Site-packages import and workflow/resource/CLI smoke | Pass |

## Focused and Integration Evidence

- Unit 1 integrated application/filesystem selection: 203 passed.
- HEN serialization and package-root boundary selection: 34 passed.
- Unit 2 exact-checkout prerequisite suite: 8 passed.
- Unit 1/Unit 2 architecture selection: 123 passed.
- Unit 3 documentation/architecture/entrypoint selection: 70 passed.
- Focused non-external OpenHENS/HEN profile: 458 passed, 4 solver tests
  deselected.
- Post-correction workspace/contracts selection: 158 passed.
- Complete fixed-seed non-solver suite: 2,181 passed, 3 optional-profile skips,
  4 solver deselections in 148.03 seconds.

## Verification Notes

The initial temporary wheel environment contained only the wheel and therefore
lacked its runtime dependencies. The environment was populated with the
declared CI runtime dependencies, and the installed-artifact smoke then passed
from site-packages. After the generic-mapping correction, the artifacts were
rebuilt, the wheel was reinstalled, and both the standard smoke and installed
mapping-key rejection probe passed. No source-checkout import was used.

The ignored local `docs/_build` cache was removed after its target and ignore
rule were verified. Documentation was built into a separate clean temporary
destination, preventing stale pages from masking source state. It is recoverable
by rerunning `scripts/build_docs.py`.

## Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Partial Property-Based Testing: enabled for generated case identifier and path
  invariants with seed `20260715` and shrinking.

## Overall Status

- Build: success.
- Tests: pass.
- Documentation: pass.
- Distribution: pass.
- Operations: N/A; no deployment or production environment change was
  requested.
- Task: complete.
