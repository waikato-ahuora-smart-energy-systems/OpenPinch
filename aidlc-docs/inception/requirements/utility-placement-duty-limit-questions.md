# Utility Placement Duty-Limit Questions

The placement service currently determines each utility duty from the targeting
cascade. An upper bound must constrain that allocation without confusing the
bound with the optimized duty written to the returned case. Please answer each
question by placing a letter after its `[Answer]:` tag.

## Question 1

How should callers specify maximum utility duties in the concise public API?

A) Add an optional `maximum_duties` mapping keyed by the globally unique
utility names, for example
`maximum_duties={"hot_iso_1": 500, "cold_iso_1": {"value": 300, "unit": "kW"}}`.
This works with generated names and names inferred from existing utilities.
(Recommended)

B) Put the maximum duty inside the existing `options` mapping, separated from
the physical arguments of `utility_placement(...)`.

C) Reinterpret each existing utility's input `heat_flow` as its upper bound;
generated-count mode would require a separate convention.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2

How should a maximum duty behave when more than one period is selected?

A) Enforce the bound independently in every selected period. A scalar applies
to every period, while an existing period-resolved value can provide different
bounds by period. (Recommended)

B) Enforce one weighted aggregate duty bound across all selected periods.

C) Accept scalar bounds only and apply the same bound independently in every
selected period.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3

What should omitted and zero maximum duties mean?

A) An omitted utility is unbounded, zero disables that utility's duty, and the
placement is infeasible when the available capped utilities cannot cover the
required heating or cooling load. Hot and cold members of a generated pair
remain independently capped. (Recommended)

B) Every utility must have an explicit positive maximum duty.

C) An omitted utility has a zero maximum duty and is therefore disabled.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A. Default utility should be available to cover any shortfall
