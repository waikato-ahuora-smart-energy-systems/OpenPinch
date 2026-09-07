# HPR changes: repeated audit and repair

Scope: all HPR numerical, graph, residual and derived-case API changes, tests,
notebook source and documentation changes in the working tree. Read independently
edited notebooks as part of the review; preserve saved outputs and unrelated work.
No commits or publication. Continue until three passes have completed or a full
pass identifies no issues. Use the existing pytest/Hypothesis stack and seed.

- [x] Pass 1: review the complete diff and new owners; reproduce boundary issues,
  add regressions, fix confirmed defects and run affected tests.
- [x] Pass 2: audit pass-1 fixes and remaining cross-module interactions;
  reproduce/fix new findings and verify them.
- [x] Pass 3: audit the resulting complete changes; fix any remaining findings,
  run final regression/coverage, notebook and artifact checks appropriate to fixes.
- [x] Record findings, validation, remaining limitations and stopping condition.

PBT remains enabled: generated period/unit transformations and target-integrity
invariants complement real HPR/placement tests. Security/Resiliency remain disabled.
No separate inception, infrastructure or deployment stages add value to this review.
