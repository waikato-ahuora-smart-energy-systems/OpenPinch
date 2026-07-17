# PinchProblem Targeting Domain Entities

## EffectiveArguments

An immutable mapping of resolved public argument names to values and provenance
labels (`named`, `options`, `config`, or `default`). It is attached to replay
intent and is not persisted back into configuration.

## TargetRunSpec

An immutable record containing the descriptive surface name, effective options,
zone scope, and subzone decision needed to replay one method for another period.

## PeriodTargetResults

An ordered collection keyed by canonical period identity. It stores independent
`TargetOutput` values and supports reporting aggregation without changing the
selected-period result.

## ConfigurationView

A read-only view of stored numerical assumptions. Persistent mutation remains
explicit through `update_options()`.

## Public Target Accessors

`TargetAccessor` owns selected-period methods. `AllPeriodsTargetAccessor` mirrors
only backend-supported methods and adds the independent `workers` control.
