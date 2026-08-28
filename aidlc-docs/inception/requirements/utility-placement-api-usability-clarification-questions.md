# Utility Placement API Usability Clarification Questions

The first answers establish short level-count names, a best-solution-only
workflow, normal-case behavior, and explicit targeting/plotting. Questions 2
and 3 did not match their original lettered options, but together they suggest
that utility placement should return a normal case which can subsequently be
added to a workspace. The following questions confirm that interpretation.

## Clarification Question 1
What should `problem.target.utility_placement(isothermal=2, sensible=2)` return?

A) A detached `PinchProblem` containing the best optimized utilities, with the
source problem unchanged. This is recommended based on the answers "only
return the best utility set" and "access should be like a normal case."

B) A plain list of best-utility input dictionaries, requiring the caller to
construct a case separately.

C) A detailed placement result containing a `best_case` property that exposes
the optimized `PinchProblem`.

D) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Clarification Question 2
How should that returned case be registered in a workspace?

A) Add a direct method:
`workspace.add(optimized_case, name="optimized_utilities", activate=False)`.
This is recommended and matches the requested `workspace.add(case)` workflow.

B) Reuse the existing load method:
`workspace.load(optimized_case, case_name="optimized_utilities",
activate=False)`.

C) Register it automatically during utility placement, without a separate
workspace call.

D) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Clarification Question 3
Where should objective value, entropy decomposition, alternatives, and
optimizer termination evidence be available?

A) Retain them on the returned normal case through
`optimized_case.utility_placement_result`; ordinary case targeting, summaries,
reports, and plots otherwise work normally. This is recommended because it
keeps engineering evidence without requiring users to manipulate it.

B) Omit placement evidence from the normal workflow and return only the best
utility definitions.

C) Return `(optimized_case, placement_result)` as a two-item tuple.

D) Other (please describe after the [Answer]: tag below)

[Answer]:  A
