# Domain Entities

## Stream

The public runtime entity uses descriptive constructor arguments, mutable state
properties, read-only derived properties, and descriptive helper field names. Private
storage remains compact to avoid changing serialization and numerical kernels at once.

## Value

`period_values` returns a defensive NumPy copy. Dictionary and JSON representations use
the established `values` key. Foreign quantity/value coercion remains supported.

## Enums

`ZoneType`, `TargetType`, `HeatExchangerNetworkDesignMethod`,
`HeatPumpAndRefrigerationCycle`, `StreamType`, `StreamID`, `ProblemTableLabel`,
`HeatExchangerNetworkLabel`, `StreamDataLabel`, and `GraphType` are the sole identities.

## HeatExchangerNetworkDesignView

The view owns a read-only `result` reference plus explicit selection, ranking, totals,
utility, and grid behavior. It performs no dynamic delegation or result serialization.

## Transport Models

`TargetInput`, stream/utility/profile/zone schemas, HEN schemas, and workspace bundles
are strict. Wire names remain unchanged and workspace versioning is mandatory.

