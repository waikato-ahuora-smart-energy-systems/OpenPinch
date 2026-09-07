# Repository audit cycle 2

## Scope and baseline

Fresh audit from committed baseline 6268d747. Inspect Value coercion and period
metadata, stream collection ownership/caches, period selection, configuration,
reporting, numerical helpers, and persistence interactions. Prior full baseline:
3180 passed, one deliberate nine-stream solver benchmark skip; all optional
notebook profiles enabled; combined coverage 96.1484%.

## Pass 1: value boundaries

Ten failures reproduced:

- Iterable type detection consumed generators before loading their magnitudes.
  Use non-consuming iteration detection and one shared predicate.
- Weight input and public views exposed shared mutable metadata, including through
  independent copies and arithmetic. Copy stored and exposed weight arrays.
- Scalar selections and summaries inherited multi-period weights, causing their
  weighted summaries to fail. Retain weights only when result shape matches.
- Mapping weights could have the wrong length and fail only during aggregation.
  Validate their length when constructing a Value.

492 domain and contract tests passed. Generated iterable/list equivalence and
concrete weighted-summary/ownership regressions cover the repairs. Existing raw
passive weight magnitudes and scalar broadcasting semantics remain supported.

## Pass 2: collections and period selection

Twenty-four failures reproduced:

- Shallow collection copies shared their membership dictionary; removing or adding
  a stream changed the original. Implement independent container state and caches,
  retaining the documented shared stream objects.
- The shallow-copy pickle fallback dropped custom sort functions. A dedicated
  shallow-copy protocol preserves live callable settings.
- Fractional and boolean indices silently selected another period, while negative
  numeric strings bypassed Value bounds. Share strict non-negative integer index
  coercion across values, scalar resolution, numeric views and stream reports.
  Python/NumPy integers and numeric strings retain support.

528 domain/contract and multi-period application tests passed. Generated collection
membership properties complement explicit invalid and supported index cases.

## Pass 3: sorting and cross-layer completion

Four failing sort regressions reproduced stale ordering after numeric changes,
name changes and changes in a callable's external state. Cache numeric sort order
against stream revisions; recompute metadata and arbitrary callable order where
no reliable revision is available. Indexed collection access and get_index now
observe the refreshed order.

Two further failures reproduced deep-copy sort loss through the original pickle
path. Deep copies now preserve live callable settings and independently copy their
streams without using the pickle fallback. Existing pickle behavior is retained.

Twelve further failures reproduced residual index truncation in stream totals,
analysis helpers and configuration. Reuse strict coercion in those paths and
segmented targeting. Configuration also rejects conflicting period IDs/indices
instead of silently ignoring the index.

622 affected domain, contract and analysis tests pass. Reviewed numerical helper
contracts, reporting/export ordering and workspace persistence interactions. No
additional confirmed repair is outstanding. All three inspection passes found
issues, so the three-pass limit is the stopping condition.

## Verification complete

The full suite with all optional tutorial profiles and fresh branch-aware coverage
passes. Application and test source remained unchanged during the run. Wheel/sdist,
installed-package smoke checks, lint/format checks and file-preservation checks
also pass. No commit or push was performed.

## Testable properties and extension compliance

- Iterable/list equivalence: generated bounded finite values, explicit [10, 20].
- Copy independence: generated collection sizes and concrete weighted metadata
  mutations; deep-copy stream isolation and shallow stream identity.
- Selection equivalence: Python/NumPy integer and numeric-string positions agree;
  invalid inputs cannot silently select a different operating period.
- Sorting oracle: after generated temperature changes, the highest-temperature
  stream must be first. Metadata and callable-state examples complement this.

PBT-01/03/05: compliant; invariants and explicit oracles identified and tested.
PBT-02/04: existing domain/contract round-trip and repeated-operation properties
retained; no new persistence format or idempotent public operation introduced.
PBT-06: mutation sequences cover independent copies and cache refresh; existing
stateful suites retained. PBT-07/08/09: bounded generators, shrinking, explicit
examples and seed 20260715; reuse Hypothesis. PBT-10: examples complement generated
properties. Security and Resiliency extensions remain disabled and are skipped.


## Packaging and preservation

Wheel and source distribution build successfully. A wheel installed into a fresh
package target outside the repository passes smoke checks for public targeting,
generator input, weighted summaries, copy independence, sort freshness and strict
period selection. All 12 changed Python files pass formatting; global Ruff and
patch whitespace checks pass. All 19 saved notebooks and the generated local
workspace file match pre-audit hashes. Post-suite hashes also pass.


## Regression inventory

64 new cases across three files: 11 Value cases; 47 collection/index/configuration
cases; six sort/deep-copy cases. Generated properties retain explicit examples.
The original domain, contract, application and integration tests remain intact.
No expected solver outputs were changed to make a test pass.

The final command is:
`OPENPINCH_TUTORIAL_PROFILES=all COVERAGE_FILE=/tmp/openpinch-audit2-full.coverage .venv/bin/python -m coverage run --branch --source=OpenPinch -m pytest --hypothesis-seed=20260715 -ra`

There are no test-name or marker exclusions. The existing deliberately skipped
nine-stream live solver benchmark remains unchanged. The interactive notebook
profile uses the existing dashboard-entry-point stubs and is not a manual browser
usability review. Final totals and coverage are recorded below.


## Final results and stopping condition

- Full suite: **3244 passed, 1 skipped**, in 1122.56 seconds. No failures or
  deselections. All optional tutorial profiles executed.
- The sole skip is the existing deliberate nine-stream live solver benchmark
  at tests/analysis/heat_exchanger_networks/test_solver_regressions.py:340.
- Fresh combined line/branch coverage: **96.1932%**, passing the 95% gate.
  Covered lines: 28815/29564; covered branches: 8406/9130. This is combined
  coverage, not branch-only coverage.
- Global Ruff, all 12 changed Python files' formatting, and patch whitespace
  checks pass. Strict documentation build/consistency checks pass in the suite.
- Wheel/sdist build and installed-package smoke checks pass.
- All 19 saved notebooks and the generated local workspace file remain unchanged.

Three audit-and-repair passes completed; each found confirmed issues. All confirmed
findings are repaired, with 64 permanent new regression cases. The three-pass
limit is the stopping condition. Repairs and verification records were subsequently committed on develop at the user's request.

Evidence logs:

- /tmp/openpinch-audit2-pass1-red.log and /tmp/openpinch-audit2-pass1.log
- /tmp/openpinch-audit2-pass2-red.log and /tmp/openpinch-audit2-pass2.log
- /tmp/openpinch-audit2-pass3-red.log
- /tmp/openpinch-audit2-pass3-deepcopy-red.log
- /tmp/openpinch-audit2-pass3-indices-red.log
- /tmp/openpinch-audit2-pass3-integrated.log
- /tmp/openpinch-audit2-full.log
- /tmp/openpinch-audit2-coverage.log and /tmp/openpinch-audit2-coverage.json
- /tmp/openpinch-audit2-build.log
