# Compatibility Shim Cleanup Clarification Questions

Question 5 grouped several behaviors that serve different purposes. The distinctions are:

- Unit-group overrides, fluid-phase normalization, compact JSON keys, and Pint or value-like coercion are current input and serialization contracts. They help process engineers express equivalent engineering data; they do not preserve retired OpenPinch APIs.
- Optional-dependency guards do not substitute old behavior. They defer imports and raise focused installation errors when an explicitly requested capability is unavailable.
- The Pyomo `available()` `TypeError` retry is dependency-version compatibility. OpenPinch now requires Pyomo 6.10 or newer for synthesis, so this is a genuine compatibility shim unless retained deliberately.
- The missing-Couenne fallback is an algorithmic resilience policy. It skips unavailable derivative or preliminary stages and continues with network evolution after issuing a runtime warning.
- The segmented-stream parent-axis zero is a current equation-shape invariant. It directs the solver to authoritative segment tensors and is not a compatibility path.

Please answer each question by adding the selected letter after its `[Answer]:` tag. Choose `X` and add a description when none of the listed options matches your preference.

## Clarification Question 1
Should the Pyomo cross-version `TypeError` retry be removed?

A) Yes; call the Pyomo 6.10-or-newer keyword API directly and fail normally if the dependency violates the supported contract (recommended)

B) No; retain the positional-call retry as an explicit dependency compatibility exception

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 2
What should happen when the configured Couenne solver is unavailable during the composite HEN workflow?

A) Retain the warning-backed algorithmic fallback to network evolution because it is resilience behavior, not API compatibility (recommended)

B) Fail fast whenever the configured solver is unavailable; never skip synthesis stages

C) Allow fallback only for the default composite workflow, while explicitly requested individual design methods fail fast

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 3
Should the current engineering-input, wire-format, optional-dependency, and solver-shape contracts remain?

A) Retain unit-group overrides, phase normalization, compact wire keys, Pint and value-like coercion, optional-dependency guards, and the segmented-stream shape invariant (recommended)

B) Retain compact wire keys and optional-dependency guards, but require strict runtime unit, phase, and value representations

C) Enforce one strict representation at every boundary, including JSON, engineering values, optional capabilities, and solver tensors

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: X - more information.
