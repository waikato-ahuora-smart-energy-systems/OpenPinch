# Unit 1 Business Logic Model

## Purpose

Unit 1 freezes the process-engineer contract and fixes result aggregation rules
that would otherwise force tutorial workarounds.

## Aggregation Model

Ordered period outputs are aligned by `(target name, row type)`. Each aligned
field follows one explicit policy:

| Policy | Fields | Rule |
|---|---|---|
| weighted | duties, work, operating cost, efficiencies, pinch values when present in every period | normalized weighted mean |
| peak design | HPR capital and annualized-capital fields | maximum compatible value |
| derived | HPR total annualized cost | weighted operating plus peak annualized capital |
| consensus | cycle, success, stream identities | shared value or `None` |
| optional diagnostic | pinch values and other nullable diagnostics | `None` when partially or wholly missing |
| aligned collection | hot and cold utilities | stable first-seen names with missing duty treated as zero |

Missing required additive/design values remain errors. Missing optional
diagnostics never abort an otherwise valid multi-period report.

## Contract Inventory Model

The target public inventory is stored as data in tests and classifies each
operation by owner, execution behavior, support level, and tutorial owner.
Root exports are exactly `PinchProblem` and `PinchWorkspace`. A later public
method addition fails the inventory guard until deliberately classified.

## State Model

The executable contract distinguishes prepared, targeted, designed, and
invalidated states. Mutations clear dependent state. Observation does not
execute. Unit 1 records these rules as failing/golden contract tests for Units 2
and 3 to satisfy.

## Testable Properties

- **PBT-02 Round-trip**: generated aligned `TargetOutput` records survive JSON
  serialization and validation before and after aggregation.
- **PBT-03 Invariant**: weighted output preserves target ordering and identity,
  produces finite values within the generated period range, and does not mutate
  inputs.
- **PBT-03 Invariant**: normalized positive weights are scale invariant.
- **PBT-07 Generator quality**: reusable strategies generate thermally and
  structurally valid aligned period outputs with optional pinch diagnostics.
- **PBT-08 Reproducibility**: Hypothesis shrinking remains enabled and CI uses
  seed `20260715`.
- **PBT-09 Framework**: Hypothesis is the selected Python framework and is
  already a locked development dependency.

PBT-01, PBT-04 through PBT-06, and PBT-10 are advisory under Partial mode;
example regressions remain alongside the enforced property tests.
