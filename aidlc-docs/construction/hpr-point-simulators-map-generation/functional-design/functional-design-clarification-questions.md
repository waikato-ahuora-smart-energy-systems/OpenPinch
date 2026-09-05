# Unit 2 Functional Design Clarification

The answer to original Question 5 was `Not selected`. Question 2 selected the
refrigerant-only TESPy loop, so source/sink circulation pumps do not exist in
the first-release model. The map still needs an explicit electrical system
boundary.

## Question 1
What should `electric_power` mean for the selected refrigerant-only TESPy
topology?

A) Compressor power only, with provenance explicitly recording that no
electrical auxiliaries are modeled. This is the recommended interpretation
consistent with the selected topology.

B) Compressor power plus a fixed auxiliary-power allowance added outside
TESPy, requiring a new explicit allowance in the generation context.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A
