# HPR derived-case API build and test summary

Status: complete. Runtime, documentation, artifacts and verification are complete;
two independently confirmed saved-output checks remain outside the final gate.

## Completed evidence

- Focused placement/API/architecture regression suite: 102 passed, seed 20260715.
- Shared lineage, ownership and compatibility: 34 passed.
- API signatures and existing energy-transfer notebook: 18 passed.
- Docs and notebook generator contracts: 21 passed.
- Real HPR child-zone/nondefault-period inference and stale-handle regression: passed.
- Solver-marked gate: 3 passed, 1 skipped (optional solver availability).
- Notebooks 08 and 19 passed in clean local Jupyter kernels and temporary working
  directories. Notebook 08 produced four nonempty HP/RF graphs (4/7/4/7 traces)
  and an optimized GCC with 17 traces. Notebook 19 cached Process/Site duties
  matched winning candidate duties and both standard plots were populated.
- All 19 notebook sources match the generator. Only 08 and 19 were updated;
  saved outputs and notebook 19 search settings were preserved.
- Strict Sphinx build passed. Wheel and source distribution built successfully.
- Installed-wheel full public API sequence and existing TESPy artifact smoke passed
  outside the checkout. No root API expansion beyond the existing two classes.
- Ruff passed for OpenPinch/tests/scripts; all 45 changed Python files passed
  formatting checks. Eleven unrelated unchanged files fail the repository-wide
  format check; no blanket formatting was applied. git diff --check passed.

## Complete-suite reassessment

An unrestricted non-solver pass ran 3,138 tests: 3,131 passed, three skipped and
four failed. Two failures were the old placement signature assertion and the
energy-transfer subzone guard, both corrected and rerun successfully. The other
two are existing saved-output invariants in test_notebooks_are_valid_nbformat_documents
and test_tutorial_review_preserves_notebook_invariants, reproduced on notebook 06.
They expect cleared execution counts/outputs. Saved user evidence was not cleared
and the checks were not weakened. The previous notebook 19 generator drift now
passes after preserving its settings in the generator.

The final complete coverage run uses finished code and separates only those two
confirmed independent notebook-output checks. The final review also corrected options-based period selection, preserving
named-selector precedence. A complete application rerun collects fresh coverage
for the three changed selection/transaction modules; older coverage for those
modules is discarded before merging. Final results: 3,134 passed, three skipped, six deselected (four solver-marked
and the two saved-output invariants), in 569.55 seconds. The final application
rerun passed all 442 tests in 270.30 seconds. These are overlapping suites and
are not summed as a unique-test count.

Fresh coverage replaced, rather than supplemented, the older measurements for
the three reviewed modules. The combined line-and-branch gate passed at
95.66376721549135 percent (threshold 95). Lines: 28,633/29,504 (97.05 percent);
branches: 8,320/9,124 (91.19 percent). The 95.66 percent figure is combined
coverage, not branch-only coverage. Reports: /tmp/hpr-api-coverage.log and
/tmp/hpr-api-coverage.json.

## Artifacts

- /tmp/hpr-api-full.log: unrestricted reassessment.
- /tmp/hpr-api-full-final.log: final full regression gate.
- /tmp/hpr-api-kernels.log: clean-kernel notebook evidence.
- /tmp/hpr-api-focused-final.log and /tmp/hpr-api-lineage.log: focused evidence.
- /tmp/hpr-api-sphinx.log, /tmp/hpr-api-build.log: documentation and package builds.
- /tmp/openpinch-api-dist: wheel and sdist; /tmp/openpinch-api-installed: smoke install.
- /tmp/hpr-api-installed-smoke-final.log and /tmp/hpr-api-artifact-smoke.log:
  installed workflow and existing artifact contracts.

## Enabled extension compliance

PBT-01: compliant; explicit properties, oracles and reference model.
PBT-02: compliant; canonical JSON/copy roundtrips.
PBT-03: compliant; frozen basis, exact scope, ownership and source preservation.
PBT-04: compliant; repeat copying and read-only observation are idempotent.
PBT-05: compliant; explicit residual workflow and canonical inputs are oracles.
PBT-06: compliant; operation sequences include empty sequences and verify state.
PBT-07: compliant; constrained period permutations, units, active/capacity values.
PBT-08: compliant; normal shrinking and fixed seed 20260715 retained.
PBT-09: compliant; existing pytest/Hypothesis stack, no new dependency.
PBT-10: compliant; real HPR/placement, components and numerical regressions
complement generated properties. Security/Resiliency remain disabled and N/A.
Separate infrastructure, deployment and performance stages are N/A; finalization
adds one deterministic allocation per selected period, with no extra search.

## Completion

All seven implementation units and applicable workflow stages are complete.
No commit, push, publication or deployment was performed. Existing saved notebook
evidence and unrelated edits were preserved. No blocking findings remain within
the implemented API scope. Separate NFR/infrastructure stages were unnecessary;
Operations is not applicable.

## Follow-up: three change audits

Final full non-solver regression: 3152 passed, 3 skipped and
6 deselected in 610.11 seconds. The six deselections are four
solver-marked tests and only the two saved-output checks described below.
All 17 new audit regressions pass within this run. Fresh combined line/branch
coverage is 95.7786% (unchanged 95% gate). Lines:
28683/29524; branches:
8345/9136. Coverage is measured from the
final source, without merging measurements from the interrupted run.

Final wheel/sdist rebuilt successfully. The installed public HPR/allocation/
placement/transfer sequence and all three new unit/period capacity cases pass
outside the checkout. Every loaded OpenPinch module was verified to come from
the installed wheel. Ruff, all 46 changed Python file formats and diff hygiene
pass. Notebook outputs and unrelated edits were preserved.

Stopping condition: three audit-and-repair passes completed as requested. All
identified implementation defects are fixed and verified; a fourth pass was not
requested. Two existing notebook-output checks still fail on notebook 06 because
they require cleared execution counts/outputs. Their assertions were not weakened
and saved evidence was not deleted. No commits, pushes or deployment performed.

Evidence: /tmp/hpr-audit-full-final.log, /tmp/hpr-audit-coverage.json,
/tmp/hpr-audit-coverage.log, /tmp/hpr-audit-kernels.log,
/tmp/hpr-audit-installed-final.log, /tmp/hpr-audit-solver.log,
/tmp/hpr-audit-saved-outputs.log. Final distributions:
/tmp/openpinch-audit-final-dist; installed copy: /tmp/openpinch-audit-installed.

Detailed findings: ../../hpr-derived-case-api/code/change-audit.md.

## Complete-suite follow-up

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

Details: ../all-tests-repair-summary.md.
