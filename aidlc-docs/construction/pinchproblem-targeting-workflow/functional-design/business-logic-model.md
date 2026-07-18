# PinchProblem Targeting Business Logic

## Execution Model

`PinchProblem` prepares input without solving. A process engineer chooses one
descriptive `problem.target.<method>()`. The method resolves effective numerical
arguments, executes only that method's service graph, stores the selected-period
result and immutable replay intent, and returns the focused target.

`problem.target.all_periods.<method>()` repeats the same named method over
canonical period identities. It retains ordered period results while restoring
the caller's selected-period state. No observation method starts analysis.

## Argument Precedence

For each supported value, precedence is:

1. explicit named keyword;
2. advanced `options` mapping;
3. stored numerical configuration;
4. method default.

An omitted sentinel distinguishes omission from explicit `False`, zero, and
`None`. Invocation values are ephemeral and never mutate stored configuration.

## Method Families

- Heat integration exposes focused direct, indirect/Total Site, explicit Total
  Site, and dependency-aware all-zone heat integration.
- HPR exposes Carnot, vapour-compression, Brayton, and MVR methods whose names
  determine the model. Placement and supported topology remain binary choices.
- Area/cost, cogeneration correlations, exergy, and energy transfer are explicit
  methods rather than configuration-selected attachments.
- All-period methods mirror supported selected-period names; unsupported
  combinations are absent.

## State Transitions

Input replacement, persistent numerical configuration updates, and component
changes invalidate derived targets, graphs, designs, reports, and period
results. Target execution populates analysis state. Summary, report, comparison,
dashboard, export, and plot methods require that state and raise actionable
errors when it is absent.
