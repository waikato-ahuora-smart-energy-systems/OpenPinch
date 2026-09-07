# 3. Selection, provenance and lifecycle rules

- Resolve the original target before constructing a derived case. Reuse
  hpr_selection validation and existing analysis provenance; use a small shared
  application resolver rather than a new analysis framework.
- Validate owner, source fingerprint, result integrity, target capability,
  physical/shifted coordinates, units, scope and period. Retained compatible HPR
  handles remain usable after a different HPR method is run on the same source.
- Omitted zone and period selectors inherit the selected target's scope/period.
  Explicit conflicting selectors fail. For a scalar HPR basis, all means all
  supported residual analyses, not every source subzone. Use sentinel defaults
  where necessary to distinguish omission from explicit include_subzones=True;
  preserve the effective ordinary defaults when no base target is supplied.
- Explicit period_ids for placement must match the frozen scalar period. Shared
  HPR aggregates and broadcasting one handle across a case/period batch are not
  supported. Reject these at the boundary; explicit per-case/per-period loops
  remain available. Existing no-base batch and all-period workflows keep working.
- Reject overrides that would change the frozen physical basis. Consumer-specific
  controls such as placement budgets and seed remain available. Do not silently
  accept a thermal setting that changes nothing or triggers process retargeting.
- Route HPR-derived calls to the derived analysis owner before entering its
  transaction. The source transaction must not accidentally commit the derived
  zone tree, reports or utility definitions back to the source problem.
- Preserve both source HPR lineage and derived-case execution provenance. A
  detached report is not automatically a locally owned target handle on the
  original problem. Do not fabricate ownership to enable unsupported chaining;
  the explicit residual or optimized case is the owner for further case analysis.
- Fail atomically. Rejected selections or unsuccessful final allocation leave
  both source studies and existing successful results intact.
