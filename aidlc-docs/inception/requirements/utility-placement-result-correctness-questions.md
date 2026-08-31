# Utility Placement Result Correctness Questions

The reproduced result exposes two separate issues. The Total Site optimizer
evaluates candidate utilities against an aggregate load profile, while ordinary
Total Site targeting first allocates utilities within child direct targets and
then aggregates them. The returned case also stores zero input heat flow and
does not preserve the optimized period-specific dispatch, so ordinary
retargeting can use different duties. One product decision remains ambiguous:
whether every requested named utility level must carry meaningful duty.

## Question 1
Which targeting behavior must define feasibility and the thermodynamic objective?

A) Use the exact ordinary Process or Total Site targeting workflow that the returned case and standard plots will run; optimizer evidence and retargeted duties must agree (recommended)

B) Keep a separate theoretical aggregate-profile optimizer even when ordinary retargeting and standard plots use different duties

X) Other (please describe after [Answer]: tag below)

[Answer]:  A

## Question 2
What does requesting two isothermal plus two sensible levels mean for level use?

A) Every named level must carry a meaningful nonzero duty in every selected period

B) Every named level must carry a meaningful nonzero duty in at least one selected period; it may be inactive in other periods

C) The levels are only available candidates and may remain unused when a lower-entropy solution needs fewer levels

X) Other (please describe after [Answer]: tag below)

[Answer]:  C

## Question 3
If meaningful nonzero duty is required, how should its lower bound be defined?

A) Add a public per-utility `minimum_duties` mapping, analogous to `maximum_duties`; omitted levels have no minimum

B) Apply a documented default minimum fraction of each period's residual side duty to every requested level

C) Do not add a lower bound because Question 2 selects candidate-only behavior

X) Other (please describe after [Answer]: tag below)

[Answer]:  N/A
