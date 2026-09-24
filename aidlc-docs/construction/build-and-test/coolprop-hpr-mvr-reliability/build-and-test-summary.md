# CoolProp HPR and MVR Reliability Build and Test Summary

Status: Complete and approved under the user's authorization through completion.

## Post-construction audit (2026-09-24)

The historical results below describe the original construction. The subsequent
`implementation-audit.md` supersedes its blanket reliability/compliance claims
and documents eleven classes of confirmed defects corrected after construction.

- New audit coverage: 23 focused regression/property cases and 14 real public
  workflow cases, all passing (37 total).
- Affected-surface run: 1,047 passed, three declared skips, one stale notebook
  generator snapshot failure. That process imported the generator before its
  final source update. A fresh notebook suite passed 25 tests with three skips;
  the exact idempotence regression also passed on isolated replay.
- Final adapter/result/search/audit boundary replay: 90 passed.
- Strict Sphinx, Ruff, formatting, compilation and patch hygiene passed.
- Final wheel and source distribution rebuilt in
  `/private/tmp/openpinch-hpr-audit-dist-20260924`.
- Installed-wheel TESPy smoke and an unmocked public CoolProp VC+MVR solve
  passed outside the checkout using the rebuilt wheel.
- Notebooks 09/11 ran successfully with non-null target/map/cascade assertions.
- A real two-worker, three-period CoolProp batch passed and exported detached
  JSON snapshots for turndown, base and peak.

### Repository-wide audit accounting

The renderer-enabled repository run collected 3,374 cases and reported 3,368
passes, four declared skips and two failures in 762.70 seconds. Both failures
used source imported before edits made during that run: the old exact-candidate
ranking assertion and the old notebook-10 generator definition. Both exact
tests passed in fresh isolated replay; the ranking suite now requires retained
original points and sorted results while permitting valid polishing improvements.

Four additional regression cases added after collection also passed: the cache
command-sequence property, two arbitrary-object rejection cases and restart
seed rotation. The final inventory is therefore 3,378 cases: **3,374 distinct
passing cases across the repository run and targeted replays, four declared
skips, and no unresolved test failure**. This is aggregate evidence, not a claim
that the final tree completed one uninterrupted all-green full-suite invocation.

Raw logs are retained under `/private/tmp/openpinch-hpr-audit-final-pytest.log`,
`/private/tmp/openpinch-hpr-audit-focused-final.log`, and
`/private/tmp/openpinch-hpr-audit-notebook-contracts.log`.
No deployment or commit is part of this audit.

## Build Status

- Build tool: Hatchling through `uv build`.
- Warning-strict Sphinx documentation: passed for all 57 source documents.
- Wheel: `openpinch-0.6.8-py3-none-any.whl`, 733 KiB.
- Source distribution: `openpinch-0.6.8.tar.gz`, 533 KiB.
- Both artifacts include the HPR reliability contracts, CoolProp preflight,
  optimizer boundary, direct-MVR implementation, and notebooks 09 through 11.
- The sandboxed first build could not resolve the isolated Hatchling requirement;
  the approved network-enabled retry completed without changing dependencies.

## Test Execution Summary

### Complete Repository Suite

- Collected: 3,341 tests.
- Passed in the aggregate run: 3,336.
- Declared skips: 4.
- The sole aggregate failure was Plotly/Kaleido being unable to launch Chrome in
  the filesystem sandbox. The exact image-export test passed with renderer
  access, producing 3,337 distinct passing tests and no product-code failure.
- Runtime: 655.41 seconds for the aggregate run; 4.56 seconds for the isolated
  renderer replay.

### Focused and Integration Evidence

- Expanded HPR/contracts/application/notebook gate: 975 passed, 3 skipped.
- Direct process-MVR, VC+MVR, and application reliability: 85 passed.
- Scalar/cascade topology and performance-map compatibility: 87 passed.
- Final changed-surface/generator replay: 45 passed.
- Packaged notebook 09 executed a successful optimized CoolProp target and
  generated a three-point detached performance map.
- Packaged notebook 11 executed direct process-MVR and proved serial/parallel
  multiperiod equivalence.

### Artifact and Static Evidence

- Installed-wheel TESPy/public-contract smoke: passed from `/private/tmp`, with
  import resolution outside the checkout.
- Ruff: passed repository-wide.
- Ruff format: all 128 affected Python files formatted.
- Python compilation: passed for package, scripts, and tests.
- Tutorial generation: idempotent; notebook source contracts passed.
- Patch whitespace: clean.

## Performance and Reliability Status

- Exact-coordinate objective caching, warm-start-first ordering, optimizer budget
  mapping, diagnostic caps, and preflight-before-search properties pass.
- Search evaluations omit engine/figure/record artifacts; accepted results are
  detached and deep-copy safe.
- No external load test applies to this local numerical library.

## Extension Compliance

- Property-Based Testing: compliant across U1-P1 through U3-P12 with fixed seeds
  and normal shrinking.
- Security Baseline: disabled; not applicable.
- Resiliency Baseline: disabled; not applicable.

## Overall Status

- Build: successful.
- Tests: successful, including the separately authorized renderer replay.
- Documentation and artifacts: successful.
- Operations: not applicable; no deployment, publication, infrastructure, or
  monitoring change was requested.
- Workflow: complete. No commit or external publication was performed.
