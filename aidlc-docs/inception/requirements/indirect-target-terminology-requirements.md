# Indirect Target Terminology and Summary Metadata Requirements

## Intent Analysis

- **Request type**: Brownfield terminology and reporting-contract correction.
- **Goal**: Represent every indirect heat-exchange result with one generic
  `IndirectIntegrationTarget`, keep immediate-subzone sums internal, and expose
  explicit scope and classification metadata instead of record names.
- **Risk**: Medium-high because the change intentionally breaks target-model,
  enum, report, comparison, notebook, and documentation contracts while
  preserving targeting mathematics.
- **Compatibility**: Clean break for Python symbols and selectors; historical
  workbook labels remain accepted only at the ingestion boundary.

## Functional Requirements

1. Rename `analysis/targeting/total_site.py` to `indirect.py` without a shim.
2. Replace `TotalProcessTarget` with internal `SubzoneAggregateTarget`, stored
   under `TargetType.SA = "Subzone Aggregate"` and excluded from reports.
3. Replace `TotalSiteTarget` with `IndirectIntegrationTarget`, stored under
   `TargetType.II = "Indirect"`, for every eligible aggregate Zone type.
4. Remove `TargetType.TZ`, `TargetType.TS`, and unused `TargetType.RT`.
5. Preserve generic `indirect_heat_integration`; restrict the Total Site
   convenience workflow to Zones whose type is `Site`.
6. Add report metadata using canonical Zone address as `scope`, Zone type as
   `zone_type`, Process or Utility as `integration_type`, and Heat Exchange,
   Heat Pump, Refrigeration, or Energy Transfer as `target_method`.
7. Remove internal record names from target-result, metric, dataframe,
   comparison, and Excel summary contracts.
8. Align multi-period rows and summary comparisons through the new metadata.
9. Update HPR, refrigeration, exergy, cogeneration, power, energy transfer,
   graphs, adapters, notebooks, and documentation to the new terminology.
10. Preserve the existing two per-Zone net-profile pairs and all targeting and
    graph numerical behavior.

## Acceptance Criteria

- Site and Process Zone indirect results are instances of
  `IndirectIntegrationTarget`; removed models and enum members are unavailable.
- The internal Site indirect record is `Site/Indirect`.
- `total_site_heat_integration` and `indirect_heat_integration` produce the same
  Site result, while the Total Site convenience method rejects non-Site scopes.
- `SubzoneAggregateTarget` remains available in Zone targeting state but is
  absent from `TargetOutput`, metrics, summaries, dashboards, and Excel rows.
- Public summary columns begin with Scope, Zone Type, Integration Type, Target
  Method, and Period ID; no Target column or target record name is exposed.
- Repeated nested names are unique because Scope uses `Zone.address`.
- Direct and indirect heat-exchange rows use Target Method `Heat Exchange`.
- Multi-period concatenation, weighted averaging, serialization, and comparison
  selection remain deterministic under the new metadata key.
- Notebook 2 targets remain approximately 180,094.613 kW hot and 82,979.376 kW
  cold, and the approximately 138.5 degC LPS SUGCC ledge remains visible.

## Extension Configuration

- Security Baseline: Disabled; no security boundary changes.
- Resiliency Baseline: Disabled; no operational changes.
- Property-Based Testing: Partial. Serialization round trips and multi-period
  metadata alignment retain fixed-seed property coverage.

## Approval

The user's supplied implementation plan and subsequent terminology correction
(`Target Method = Heat Exchange`) explicitly approve these requirements.
