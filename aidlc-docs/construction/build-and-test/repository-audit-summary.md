# Repository-wide audit and repair

## Review scope and baseline

Risk-focused review across application ownership and transactions, input contracts
and adapters, domain/value/problem-table operations, numerical optimisation and
candidate handling, plotting, persistence, packaging and CI. This is not a claim
of formal correctness of every equation or third-party solver. Prior baseline:
3160 tests passed, all tutorial profiles ran, one deliberately skipped benchmark.
Saved notebook evidence and existing working-tree edits are preserved.

## Pass 1

Confirmed a mutable-input ownership defect: TargetInput loads retained the caller's
model and validate() returned the stored model. Editing either changed canonical
input without rebuilding prepared streams or invalidating cached results. Two
failing reproductions established the bug; equivalent mapping cases passed.

Copy models at source loading and validate a detached input snapshot. Update the
old private loader identity assertion to require equal but independent input.
128 affected public-input, validation and reliability tests pass.

Reviewed source rollback, JSON/workspace adapters and public result/config views;
no further confirmed defects in those inspected paths. CSV coercion and candidate
normalisation warrant concrete second-pass reproductions.

## Pass 2

Confirmed two independent defects with three failing reproductions:

- CSV/Excel unit-bearing numeric columns silently converted malformed text to
  missing values, allowing default prices or temperatures to replace bad input.
  Reject malformed values with column and data-row context; retain real blanks.
- Candidate clustering added a constant to all coordinate ranges when any one
  coordinate was fixed. This merged distinct solutions in small varying ranges.
  Normalise varying coordinates independently and use a neutral fixed-axis scale.

123 adapter, optimisation and ownership tests pass. A generated regression checks
that adding a fixed coordinate cannot alter candidate clustering. The package and
lockfile versions match. Reviewed release workflows and resource handling without
additional confirmed defects.

## Pass 3

Nine new regressions reproduced problem-table boundary failures:

- Selecting all-missing or empty columns raised StopIteration.
- Integer-backed buffers truncated fractional updates and could corrupt inserted
  missing values. Construct numeric tables with floating-point storage.
- Empty mapping/list inputs crashed or produced a one-dimensional buffer.
  Preserve a two-dimensional shape, including zero-row and zero-column tables.
- Padding list input appended columns to the caller's list. Copy its outer list
  before padding.

352 domain and problem-table tests pass. Generated cases check preservation of
fractional updates and inserted missing values for integer input in both supported
input forms. Reviewed repair interactions with interval interpolation, candidate
normalisation, shared workbook parsing and typed-input ownership. Global Ruff and
patch whitespace checks pass.

All three inspection passes are complete; each found and repaired issues, so the
three-pass limit is the stopping condition. Final integrated verification is complete with all optional tutorial profiles,
no filters and fresh branch-aware coverage. Application source remained unchanged
during the run. All confirmed findings were repaired.


## Testable properties and extension compliance

- Input ownership invariant: external mutations leave canonical input and cached
  results unchanged. Concrete mapping/model cases exercise both load and validate.
- Candidate clustering invariant: appending a fixed coordinate preserves clusters.
  Hypothesis varies the active range, with the shrunk 1e-8 example retained.
- Table numeric oracle: fractional updates and insertions preserve the specified
  values and missing entries for integer input. Generated integers cover both
  mapping and list forms; the zero example is retained explicitly.
- Missing-column shape invariant: selecting a missing column preserves row count.
  Explicit empty, single-row and multi-row cases complement existing domain PBT.

PBT-01/03/05: compliant; the invariants and numerical oracles above are covered.
PBT-02/04: compliant through existing input/domain round-trip and repeated-operation
properties; no new serialisation format or idempotent public operation introduced.
PBT-06: existing stateful and lifecycle suites retained; detached ownership cases
exercise mutate-after-load and mutate-after-validate sequences. No new state machine.
PBT-07/08/09: compliant; bounded finite generators, Hypothesis shrinking, retained
explicit examples and seed 20260715; no new testing dependency.
PBT-10: compliant; 20 concrete/generated regression cases complement the full suite.
Security and Resiliency extensions remain disabled and are skipped.

## Packaging and preservation

Source distribution and wheel build successfully. A wheel installed into a clean
package target outside the repository passes smoke checks for public targeting,
detached input ownership, numeric-text rejection, clustering and problem-table
precision/missing values. All 19 saved notebook SHA-256 hashes match the pre-audit
baseline. Changed-file formatting passes. No commits, pushes or deployments.


## Final verification

Command: `OPENPINCH_TUTORIAL_PROFILES=all COVERAGE_FILE=/tmp/openpinch-repo-audit.coverage .venv/bin/python -m coverage run --branch --source=OpenPinch -m pytest --hypothesis-seed=20260715 -ra`

Result: **3180 passed, 1 skipped**, in 1197.31 seconds. No failures or deselections.
All base and optional slow-hpr, solver and interactive tutorial profiles ran. The
single skip is the existing deliberately disabled nine-stream live solver benchmark
at tests/analysis/heat_exchanger_networks/test_solver_regressions.py:340.
The interactive profile verifies notebook execution with dashboard entry points
stubbed by the existing test harness; it is not a manual browser usability review.

Combined line/branch coverage: **96.1484%**, passing the 95% gate. Covered lines:
28767/29527; covered branches: 8403/9132. This percentage is combined coverage,
not branch-only coverage. The 20 new audit regressions also pass in a focused run,
including explicit shrunk examples added to the generated numerical tests.

Global Ruff, changed-file formatting and patch whitespace checks pass. Strict
documentation builds and consistency checks pass as part of the full suite.
Post-run hashes again confirm all 19 saved notebooks are unchanged.

Evidence logs:

- /tmp/openpinch-repo-pass1-red.log and /tmp/openpinch-repo-pass1.log
- /tmp/openpinch-repo-pass2-red.log and /tmp/openpinch-repo-pass2.log
- /tmp/openpinch-repo-pass3-red.log and /tmp/openpinch-repo-pass3.log
- /tmp/openpinch-repo-regressions-final.log
- /tmp/openpinch-repo-audit-full.log
- /tmp/openpinch-repo-audit-coverage.log and /tmp/openpinch-repo-audit-coverage.json
- /tmp/openpinch-repo-audit-build.log

Stopping condition: three audit-and-repair passes completed. Each pass found
issues; all confirmed findings are fixed and the final integrated checks pass.
No further audit pass or pending repair remains within the requested scope.
