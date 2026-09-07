# Complete test-suite repair

## Confirmed issue and correction

Two notebook tests required all notebooks except 19 to have empty outputs and
execution counts. This contradicted the existing generator, which intentionally
preserves valid saved execution evidence, and the approved requirement to retain
independent notebook edits. All 19 files passed Jupyter schema validation.

Saved notebook checks now use nbformat.validate, retaining the existing tutorial
structure and generated cell-identity assertions. Fresh generation is still
required to have no execution counts or outputs, checked for every generated
notebook. Existing byte-for-byte generator preservation checks remain in place.
No saved notebook outputs were deleted and no application behavior was changed.

Evidence: the two original failures were reproduced; all four affected validation,
review-property, fresh-generation and preservation tests pass. Ruff and diff
hygiene pass.

## Complete run

Command: `OPENPINCH_TUTORIAL_PROFILES=all COVERAGE_FILE=/tmp/openpinch-alltests.coverage .venv/bin/python -m coverage run --branch --source=OpenPinch -m pytest --hypothesis-seed=20260715 -ra`

No test-name or marker filters. Slow HPR, solver and interactive tutorial profiles
are enabled. Local browser process access is enabled for Chrome image export.
The repository's deliberately skipped nine-stream solver benchmark is unchanged.
Complete result: 3160 passed, 1 skipped, no failures and no
deselections, in 1160.91 seconds. All three optional tutorial profiles ran.
The one skip is the repository's explicitly disabled nine-stream live solver
benchmark, not a missing dependency or a failure hidden by this repair.

Fresh combined line/branch coverage: 96.1459% (95% gate).
Lines: 28764/29524; branches:
8406/9136. Ruff, changed-file formatting
and diff hygiene pass. SHA-256 checks confirm all 19 saved notebooks are byte-for-
byte unchanged. No application implementation changes were needed in this follow-up.

Logs: /tmp/openpinch-alltests.log, /tmp/openpinch-alltests-coverage.log,
/tmp/openpinch-alltests-coverage.json and /tmp/openpinch-alltests-notebook-fix.log.
The previously reported two notebook-state test failures are resolved.
No commits, pushes or deployments performed.


## Extension compliance

PBT-01/03/05: compliant; schema, review-layout and cell-identity invariants have
explicit oracles. PBT-02/04/06: existing JSON round-trip, generator repeatability
and no-rewrite tests remain; no application transformation added. PBT-07/08/09:
existing constrained tutorial-name generation, shrinking and seed 20260715
retained, with no new dependency. PBT-10: concrete packaged notebooks and
execution tests complement properties. Security and Resiliency remain disabled.
