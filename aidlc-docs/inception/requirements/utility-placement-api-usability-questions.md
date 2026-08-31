# Utility Placement API Usability Questions

The current numerical behavior is retained. These questions only define the
public Python user journey. Security and Resiliency remain disabled, while the
existing Property-Based Testing extension remains enabled.

## Question 1
How should users specify the number and kind of utility levels on each hot and
cold side?

A) Use short, explicit counts: `utility_placement(isothermal=2, sensible=2)`.
This is recommended because the thermodynamic meaning stays visible without
the repetitive `_level_count` suffix.

B) Use a total and a sensible subset:
`utility_placement(levels=4, sensible=2)`, deriving two isothermal levels.

C) Use only a total count: `utility_placement(levels=4)`, with OpenPinch
choosing the isothermal/sensible split from a documented default.

D) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 2
What should be the primary way to turn a placement into a new named workspace
case?

A) Let the workspace create it explicitly:
`workspace.create_case_from_placement(placement, source_case="baseline",
case_name="optimized_utilities", period_id="0")`. This is recommended because
the method clearly creates a case, leaves the source unchanged, and hides all
utility-dictionary construction.

B) Let the result create it:
`placement.create_case(workspace, source_case="baseline",
case_name="optimized_utilities", period_id="0")`.

C) Combine optimisation and case creation in one target call, such as
`problem.target.utility_placement(..., case_name="optimized_utilities")`.

D) Other (please describe after the [Answer]: tag below)

[Answer]: C. Only return the best utility set. Then allow workspace.add(case).

## Question 3
Where should normal result inspection live?

A) On the returned placement: `placement.metrics`,
`placement.summary_frame()`, `placement.report()`, and
`placement.utilities(period_id="0")`. This is recommended because the value a
user receives is also the value they inspect and reuse.

B) Keep inspection on the problem:
`problem.utility_placement_summary_frame(placement)` and related methods.

C) Return only plain nested data and leave conversion/reporting to users.

D) Other (please describe after the [Answer]: tag below)

[Answer]:  Access should be like a normal case.

## Question 4
How much work should creating the optimized case perform automatically?

A) Create and register the new case with optimized utilities only; targeting
and standard GCC/TSP plotting remain explicit. This is recommended because it
avoids hidden analysis while reducing the current boilerplate.

B) Create the case and run the matching direct or Total Site target, but leave
plotting explicit.

C) Create the case, run both targets, and return the standard GCC and TSP
figures automatically.

D) Other (please describe after the [Answer]: tag below)

[Answer]: A
