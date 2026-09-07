# HPR notebook observations: verification

Status: complete. Implementation, documentation, Build and Test and workflow
records are complete under the user's explicit authorization through completion.

## Delivered behavior

Heat-pump condenser duty is bounded by the selected heating ceiling. Unused
low-grade heat incurs no HP feasibility penalty. Fraction inputs are finite in
[0, 1]; selected and achieved duties, cycle work and ambient exchanges are exposed
through hpr_load. Global economic defaults are unchanged; notebook 08 uses a
positive cold/electricity ratio of 0.1 and explains price sensitivity.

HP and refrigeration produce distinct, finite GCC and net-load graphs. Plotting
is read-only and supports explicit target handles, rejecting stale/foreign or
modified handles. Residual postprocessing retains exact breakpoints and ambient
exchange for both process and utility-system HPR. Segmented utility shape and
maximum capacity survive detachment.

```python
residual = problem.target.residual_utility(base_target=heat_pump)
utilities = residual.target.direct_heat_integration()
optimized = residual.target.utility_placement(isothermal=2)
optimized_utilities = optimized.target.direct_heat_integration()
```

Placement evaluates the frozen residual, preserving its original HPR duty,
temperatures, physical boundary, selected period and provenance. It is sequential
optimization; it does not resize HPR. Shared-vector multiperiod HPR aggregates
are excluded from conversion. No commit, push or deployment was requested.

## Verification evidence

- Full non-solver suite before the final indirect-ambient correction: 3,112
  passed, 3 skipped, 7 deselected in 530.47 seconds. This comprises four normal
  solver deselections and the three independent notebook checks described below.
- Solver-marked gate: 3 passed, 1 skipped, 3,115 deselected in 102.12 seconds.
- Focused graph/accounting/lifecycle correction gate: 68 passed in 25.04 seconds.
- Default and explicit live graph selection: 1 passed in 14.43 seconds.
- Final post-ambient HPR, architecture and public-contract gate: 731 passed in
  145.60 seconds.
- Final aggregate suite after all code corrections: 3,116 passed, 3 skipped and
  7 deselected in 532.56 seconds. Combined line/branch coverage is 95.75177%
  (28,524/29,366 statements and 8,260/9,050 branches covered); the repository
  coverage report --fail-under=95 gate passed without changing its threshold.
- Strict Sphinx build, Ruff lint/changed-file format, patch whitespace, wheel and
  sdist builds: passed. The final warning-strict guide rebuild also passed.
- Notebook 08: nbformat validation and generator equivalence passed. A fresh
  local Jupyter kernel executed every code cell successfully in a temporary
  directory; the packaged source notebook remains unexecuted.
- Notebook plots: HP NLP 4 traces, HP GCC 7, RF NLP 4, RF GCC 7. Example HP heating
  187.500 kW, work 21.851 kW; refrigeration cooling 250.000 kW, work 447.546 kW.
  HP residual allocation: 562.500 kW hot and 834.351 kW cold.
- The installed-wheel artifact smoke with the existing TESPy dependency surface
  passed outside the checkout. A core-only invocation correctly rejected the
  shared environment because TESPy was already installed; no dependency was
  removed to make that invocation pass. The final rebuilt-wheel artifact replay
  and indirect RF residual conversion, JSON roundtrip, allocation and placement
  all passed. Six key HPR code/notebook artifacts are byte-identical in source,
  wheel and sdist.
- A supplemental utility-system RF case exposed omitted ambient heat: allocation
  and physical composites differed by 14.011 kW. After correction, remaining
  duties are approximately 0.003 kW hot and 750.010 kW cold, and real placement
  succeeds. The permanent regression verifies physical balance and frozen basis.

## Existing working-tree limitations

An earlier unfiltered run returned 3,106 passed, 6 failed, 3 skipped and 4
solver deselections. Two obsolete fixture/document expectations were corrected.
The image-export failure was caused by sandboxed Chrome startup and passes with
local renderer access. Three remaining failures belong to independent user
notebook edits and were preserved:

1. test_notebooks_are_valid_nbformat_documents expects cleared execution counts;
   notebook 06 has saved execution evidence.
2. test_tutorial_review_preserves_notebook_invariants enforces the same cleared
   execution invariant on that notebook.
3. test_notebook_generator_does_not_rewrite_current_notebooks finds independently
   edited notebook 19 source differs from its generator.

The final aggregate command excludes exactly those three test names. No test was
deleted, skipped in source, or weakened to conceal those independent changes.
Only notebook 08 was regenerated. The user's other notebooks and untracked
openpinch-workspace.json remain untouched.

## Extension and completion scope

PBT-01 through PBT-10 are compliant; detailed property evidence is in each unit's
code summary and functional design. Hypothesis seed 20260715 matches CI; shrinking
remains enabled. Properties cover finite profiles, load/energy invariants,
independent cascade offsets, repeated normalization, graph JSON transport and
random detached-case operation sequences compared with a frozen reference.
Example regressions cover actual solver, plots, allocation, placement and errors.
Security/Resiliency are disabled and skipped. Dedicated performance and
infrastructure stages are N/A under the approved workflow. Operations introduces
no work for this library change.

## Completion checklist

- [x] Four implementation units and their design/code plans are complete.
- [x] Regression, property, public-workflow and final aggregate checks completed.
- [x] Notebook, generated inventories, documentation and distributions verified.
- [x] Installed-wheel HPR and residual placement smoke completed.
- [x] All applicable enabled extension rules assessed and compliant.
- [x] Existing unrelated notebook edits preserved and exclusions disclosed.
- [x] Workflow state and append-only audit updated.
- [x] Operations assessed as N/A; no deployment or publication requested.
