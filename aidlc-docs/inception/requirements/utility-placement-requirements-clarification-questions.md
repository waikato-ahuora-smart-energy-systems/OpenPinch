# Utility Placement Requirements Clarification Questions

The initial answers are complete, but sensible utilities, cogeneration, and
shared multiperiod placement require four linked definitions before the
mathematical contract is unambiguous.

## Question 1
For each sensible utility level, how should its two endpoint temperatures be parameterised?

A) Optimise supply and target temperatures independently, creating two temperature decision variables per level

B) Optimise one placement temperature per level and apply a caller-supplied or configured fixed temperature span to derive the other endpoint (recommended)

C) Optimise the supply temperature while using one fixed return temperature shared by all hot levels and another shared return temperature for all cold levels

X) Other (please describe after the [Answer]: tag below)

[Answer]: A; the decision variables should be supply temperature and delta temperature for the utility. For isothermal utility, the delta temperature is fixed.

## Question 2
How should entropy generation be calculated for sensible utilities?

A) Integrate the constant-heat-capacity utility entropy change between its endpoints and the matched process-side entropy change over the allocated temperature intervals (recommended)

B) Treat each sensible utility as an equivalent isothermal level at its logarithmic-mean absolute temperature

C) Calculate only the external utility exergy requirement and derive entropy cost from that value and ambient temperature

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 3
How should price and cogeneration identity be attached to utility levels when technology identity is not an optimisation variable?

A) Use a fixed caller-supplied template for each level containing its hot or cold role, temperature span, price, fluid metadata, and cogeneration eligibility; optimise only its placement temperature (recommended)

B) Use temperature-dependent price functions and treat every eligible hot level as steam for the existing turbine model

C) Use one global hot-utility price and one global cold-utility price, and treat every hot level as steam for cogeneration

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 4
Which rule takes precedence for the utility temperature span (`delta_temperature`)?

A) The custom Question 1 answer takes precedence: each template fixes role, price, fluid metadata, and cogeneration eligibility but supplies bounds or an optional fixed value for `delta_temperature`; sensible levels optimise supply temperature and `delta_temperature`, while isothermal levels fix `delta_temperature` to zero (recommended)

B) The selected Question 3 wording takes precedence: every template fixes `delta_temperature`, so only supply or placement temperature is optimised

C) Each template explicitly chooses whether `delta_temperature` is fixed or optimised, allowing mixed fixed-span and variable-span levels in one solve

X) Other (please describe after the [Answer]: tag below)

[Answer]: A. However, isothermal levels fix `delta_temperature` to near zero, there is a default value.

## Question 5
How should a shared utility placement be evaluated across multiple operating periods?

A) Require the same placement to be feasible in every period and minimise the existing period-weighted sum of objectives (recommended)

B) Build one weighted-average process profile and optimise against that aggregate profile only

C) Permit period-level infeasibility and include a configurable infeasibility penalty in the weighted objective

X) Other (please describe after the [Answer]: tag below)

[Answer]: A
