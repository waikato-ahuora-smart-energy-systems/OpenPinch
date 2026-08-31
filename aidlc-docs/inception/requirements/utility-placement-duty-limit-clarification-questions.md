# Utility Placement Duty-Limit Clarification Questions

The new answer requests a default utility to cover shortfall, while the earlier
approved thermodynamic requirements reject any positive default utility duty.
Please answer each question by placing a letter after its `[Answer]:` tag.

## Question 1

When capped named utilities cannot cover a period's required duty, how should
the default utility participate?

A) Add the default utility only as residual fallback after applying the named
utility caps. It is not one of the requested or inferred placement levels, but
it is included in the returned optimized case and standard plots whenever its
allocated duty is positive. (Recommended)

B) Use the default utility to quantify the shortfall, but keep any positive
default duty infeasible and exclude it from the returned case.

C) Treat the default utility as an ordinary unbounded placement level that may
receive duty before capped named utilities.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2

How should the optimizer discourage use of a default fallback utility?

A) Rank candidates lexicographically: first minimize total period-weighted
default duty, then minimize physical entropy generation among candidates with
the same minimum fallback duty. This prevents a unit-dependent scalar penalty
from being overwhelmed by entropy differences. (Recommended)

B) Add a configurable scalar penalty multiplied by default duty to the entropy
objective.

C) Apply no extra penalty; include default utility entropy on the same basis as
named utilities.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: Use a squared penalty function, g_penalty(), of the default utility
