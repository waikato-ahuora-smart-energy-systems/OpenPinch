# Maximum-Duty Domain Entities

## Utility duty limit

A frozen value identifies one globally unique utility and contains canonical
non-negative period limits aligned to selected period IDs. Absence of an entity
means unbounded duty; zero is a meaningful explicit limit.

## Runtime maximum heat flow

An optional period-aware property on a utility stream. The targeting cascade
reads this as capacity and continues to write the allocated duty to
`heat_flow`; the two values are never conflated.

## Fallback utility allocation

A generated `HU` or `CU` allocation with a stable side, near-isothermal
temperature definition, period duty, and fallback marker. It is present in
period results and returned utility inputs but absent from placement decision
coordinates and requested level counts.

## Fallback penalty evidence

Each period exposes a dimensionless non-negative squared penalty and the
candidate exposes its raw period-weighted aggregate. Physical entropy fields
retain their existing units and decomposition.
