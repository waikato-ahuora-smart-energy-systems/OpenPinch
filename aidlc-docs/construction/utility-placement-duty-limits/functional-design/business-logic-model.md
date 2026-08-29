# Maximum-Duty Business Logic Model

## Input normalization

Resolve generated or inferred utility identities first, then normalize the
optional `maximum_duties` mapping against the final globally unique names.
Convert each scalar or period-resolved value to the request heat-flow unit and
materialize one non-negative optional limit per utility and selected period.

## Candidate allocation

For each period, create fresh named utility streams carrying their period cap.
The existing temperature-priority targeting cascade assigns no more than the
remaining cap to each named stream. An omitted cap is infinite and a zero cap
assigns zero duty. After all named streams have been considered, a generated
near-isothermal `HU` or `CU` stream supplies only the residual side demand.

The fallback hot utility is placed at the upper shifted profile boundary and
the fallback cold utility at the lower shifted profile boundary, using the
configured near-isothermal span. These definitions guarantee reachability
without becoming placement decision coordinates.

## Penalty and objective

For each period:

`g_penalty[p] = (Q_HU[p] / Q_heat_required[p])^2 + (Q_CU[p] / Q_cool_required[p])^2`.

A side contributes zero when required and fallback duties are both zero;
positive fallback against zero required duty is invalid.

Aggregate with the raw period weights. Combine the aggregate dimensionless
penalty monotonically with the entropy-ranking scalar while preserving the
feasible/infeasible scalar partition. Retain physical
entropy generation and exergy destruction unchanged in their physical units.

## Returned normal case

Write every optimized named utility with its optional period-aware maximum-duty
metadata. Write `HU` or `CU` when fallback is positive in any selected period.
Ordinary targeting reads the runtime maximum duty, respects it in the same
temperature-priority cascade, and allocates the explicit fallback residual.
The source problem remains unchanged.

## Property inventory

- allocation never exceeds a finite cap;
- scalar limits broadcast and period values select by identity;
- zero caps disable and omitted caps do not constrain;
- hot/cold generated pair caps remain independent;
- named duty is assigned before fallback duty;
- `g_penalty()` is non-negative, zero iff fallback is zero, monotone, and
  invariant under common scaling of fallback and required duties;
- weighted aggregation equals a direct period sum;
- request/result JSON and returned-case JSON round trips preserve limits;
- repeated fixed-seed execution is deterministic and source-state invariant.
