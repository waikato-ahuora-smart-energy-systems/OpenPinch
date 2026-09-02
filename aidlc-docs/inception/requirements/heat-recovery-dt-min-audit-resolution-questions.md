# Heat-Recovery DT_MIN Audit Resolution Questions

Please answer every question by placing the selected letter after its
`[Answer]:` tag. Recommended choices are identified explicitly.

## Question 1
Which input shapes should `heat_recovery` accept?

A) Accept only non-Boolean numeric scalars, OpenPinch `Value` scalars, Pint scalar quantities, and exact `{"value": number, "unit": string}` scalar mappings. Reject numeric strings, bytes, sequences, NumPy arrays, arbitrary mappings, and nested Booleans. Apply equivalent strict checks to the numerical contributor functions. (Recommended)

B) Retain permissive numeric strings and one-element sequences, but reject Boolean values and arbitrary mappings.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2
What numerical accuracy contract should apply at recovery-curve boundaries?

A) Preserve the documented `1e-6 delta_degC` contract and change the evaluator or boundary procedure so the returned physical boundary is accurate and post-verified at that tolerance, including coincident-temperature plateaus. (Recommended)

B) Preserve the existing cascade behavior and change the public tolerance to `2e-5 delta_degC` or another empirically safe value.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3
How should a supplied `Zone` object be handled?

A) Resolve its address against the current problem's execution root and use that problem's local zone; reject it if the address does not exist. This also makes a zone object safe across workspace cases that share the same hierarchy. (Recommended)

B) Accept a `Zone` only when it is the exact object owned by the current problem; require a string address for workspace batches and foreign problems.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 4
How should a positive recovery request at or below the `1e-6 kW` absolute comparison tolerance behave?

A) Reserve `zero_recovery_boundary` for an exact zero request. Treat every positive request as `solved` and require the returned point to achieve at least the positive request, using a stricter comparison for this micro-duty branch. (Recommended)

B) Reject positive requests at or below `1e-6 kW` as being below the service's numerical resolution.

C) Keep the current numerical result but document that sub-tolerance positive requests are treated as zero-equivalent and receive `zero_recovery_boundary`.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 5
How strict should direct construction or JSON deserialization of `HeatRecoveryDtMinResult` be?

A) Enforce finite numeric values without string coercion, delta-temperature units for `dt_min`, heat-flow units for recovery fields, non-negative `dt_min` and recovery quantities, and tolerance-aware relationships among requested, achieved, limit, and residual. Keep JSON round trips valid. (Recommended)

B) Enforce strict numeric types and dimensional units, but leave relationships among result fields unchecked.

C) Treat the result as a serialization container and only correct the documentation so semantic validity is guaranteed solely for service-generated instances.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 6
When an all-period mapping's keys exactly match canonical period IDs that happen to be `value` and `unit`, which interpretation should take precedence?

A) Exact canonical period-ID matching takes precedence; only mappings that do not exactly match the periods are considered scalar value-with-unit representations. (Recommended)

B) Reserve `value` and `unit` as prohibited period IDs and reject such problem inputs during validation.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A
