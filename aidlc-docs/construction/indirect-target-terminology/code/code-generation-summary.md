# Indirect Target Terminology Implementation Summary

## Outcome

OpenPinch now represents every utility-mediated heat-exchange result with
`IndirectIntegrationTarget` under `TargetType.II = "Indirect"`. The immediate
subzone sum is an internal, non-reportable `SubzoneAggregateTarget` under
`TargetType.SA = "Subzone Aggregate"`. The old module, runtime classes, and
TS/TZ/RT enum members have been removed without compatibility shims.

## Public Contract

- Runtime target record names remain private and use canonical scope, including
  `Site/Indirect`.
- Reportable targets expose Scope, Zone Type, Integration Type, and Target
  Method. Heat-exchange routes use `Heat Exchange`.
- Target results, metrics, compact and detailed summaries, comparisons,
  weighted-period alignment, workspaces, batches, dashboards, and Excel
  summaries no longer expose record names.
- Total Site convenience workflows validate Site scope; generic indirect
  targeting remains available for Process Zone, Site, Community, and Region.
- Historical Total Site and Total Process workbook labels are normalized only
  by the tabular input adapter.

## Preserved Behavior

The targeting mathematics, graph coordinate schemas, and both per-Zone
net-profile pairs are unchanged. Notebook 2 retains Total Site targets of
approximately 180,094.613 kW hot and 82,979.376 kW cold, corrected hierarchy
profile duties, and the LPS ledge near 138.5 degC.

## Extension Compliance

- Security Baseline: N/A because it is disabled for this unit.
- Resiliency Baseline: N/A because it is disabled for this unit.
- Partial Property-Based Testing: compliant through fixed-seed serialization
  round trips and metadata-based multi-period alignment coverage.
