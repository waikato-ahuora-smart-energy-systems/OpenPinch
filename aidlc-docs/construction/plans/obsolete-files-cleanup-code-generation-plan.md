# Obsolete files cleanup

## Requirements and workflow

User request: "Clean up obsolete and old files. "

Remove demonstrably obsolete artifacts without changing supported behavior.
This is one low-risk maintenance unit. Existing architecture records establish
the package boundaries; direct reference checks establish cleanup decisions.
Workflow: workspace inspection, minimal requirements, cleanup plan, removal,
and focused verification. Stories and application, functional, NFR, unit, and
infrastructure design are unnecessary because no application behavior changes.
The cleanup request authorizes the reversible repository changes below.

## Removal evidence

- `coverage.json`: generated 2026-06-27; no current code, CI, or documentation
  consumes it. The existing code-quality assessment also identifies it as waste.
- `examples/debug_support_notebook.ipynb`: unreferenced scratch notebook using
  retired `use_case`, `copy_case`, and `set_dt_cont_multiplier` APIs and a
  non-package `files("stream_data")` resource lookup. Maintained tutorials live
  in `OpenPinch/data/notebooks` and have a generator and coverage inventory.
- Local Python, pytest, Ruff, and Sphinx caches, Finder metadata, root coverage
  database, old timing log, and the two local 0.6.3 release archives: reproducible
  artifacts. The current project release is 0.6.5.

Preserve all workbook versions, input fixtures, historical result baselines,
research and private data, environments, OpenHENS checkout, Hypothesis examples,
workflow history, and the user's modified utility-placement notebook.

## Execution checklist

- [x] Step 1: Inspect tracked/untracked state, consumers, packaging, and workflow.
- [x] Step 2: Record exact cleanup scope and capture modified-notebook checksum.
- [x] Step 3: Remove confirmed artifacts; ignore root generated coverage JSON.
- [x] Step 4: Run entrypoint, architecture, resource, and example checks; verify
  ignore behavior, patch hygiene, removal manifest, and notebook preservation.
- [x] Step 5: Record verification and finish state tracking.

## Extension compliance

Security and Resiliency are disabled in project state and skipped.
PBT-01 through PBT-07 and PBT-10: N/A; no business logic, contracts, or tests
change. PBT-08: retain existing seed-based execution. PBT-09: compliant;
Hypothesis remains in the development dependencies. No blocking findings.

## Second pass: retired workbook result snapshots

The repeated cleanup request authorizes a deeper pass. The original retention
decision is narrowed for these 55 confirmed unused generated outputs:
`examples/results/r_*.json` (54 files) and
`examples/review/other/r_new_example_3.json`. All have the older workbook target
schema. Current tests solve the 54 input fixtures directly; the architecture
migration baseline explicitly says these files are not its regression oracle.
Keep source workbooks, every input fixture, and active specialist baselines.
Keep workflow records because they document decisions and verification history.

- [x] Step 6: Check output consumers, schema, migration baseline and source files.
- [x] Step 7: Remove the 55 generated outputs; retain regeneration by creating
  its results directory; ignore regenerated review result JSON.
- [x] Step 8: Verify example workflows, workbook adapter, generator directory
  recreation, fixture preservation, and lint; update the existing summary.

No answer was received to the optional Excel v4.02 question during this pass.
Both historical workbook releases are retained. Existing extension configuration
is retained. This small filesystem repair adds no numerical or contract logic;
verify it with a controlled temporary-directory smoke check.
