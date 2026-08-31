# Indirect Profile Precision Requirements

## Intent Analysis

- **Request type**: Brownfield numerical bug fix.
- **Scope**: Direct target graph extraction and indirect profile reconstruction.
- **Complexity**: Simple implementation with medium scientific-correctness risk.
- **Compatibility**: Preserve public APIs, graph rounding, target schemas, and
  existing coarse-resolution results.

## Functional Requirements

1. Preserve the full-precision shifted Direct Integration cascade on
   `DirectIntegrationTarget.pt` after graph data is generated.
2. Continue rounding direct graph payloads to four decimal places.
3. Reconstruct immediate-subzone indirect profiles from the full-precision
   `T` and `H(net)-actual` columns of each child Direct Integration target.
4. Do not round reconstructed segment temperatures before the indirect cascade
   is calculated.
5. Preserve enthalpy duties and temperature intervals that differ only beyond
   four decimal places.
6. Preserve the two independent per-Zone net-profile pairs and the no-child
   fallback.

## Acceptance Criteria

- Direct graph generation does not mutate either supplied Problem Table.
- Direct graph arrays retain their existing four-decimal presentation.
- A child interval whose endpoints collapse under four-decimal rounding remains
  distinct during indirect reconstruction.
- Reconstructed duties retain sub-four-decimal enthalpy precision.
- Existing Total Site hierarchy, Notebook 2, graph, multi-period, and HPR tests
  remain green.

## Extension Compliance

- Security Baseline: N/A, disabled.
- Resiliency Baseline: N/A, disabled.
- Partial Property-Based Testing: applicable to reconstruction precision and
  duty-conservation invariants.
