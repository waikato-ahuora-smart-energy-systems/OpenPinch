# Business Logic Model

`PinchProblem.components` mutates prepared process data and invalidates all
derived results without solving. `PinchProblem.design` selects one named HEN
algorithm, establishes its fixed heat-integration prerequisite, and returns an
application view over serializable synthesis data. `PinchProblem.plot` and all
report/export operations consume existing results only.

`PinchWorkspace.scenario` clones stored canonical input into an unsolved named
problem. `PinchWorkspace.cases` returns an insertion-ordered batch whose target
and design methods mirror the single-problem vocabulary and report per-case
outcomes deterministically.
