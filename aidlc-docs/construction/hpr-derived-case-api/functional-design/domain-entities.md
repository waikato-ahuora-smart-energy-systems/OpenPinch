# 4. Optimized-case finalization

The optimizer already evaluates candidate allocations, but the current
application function reconstructs a case with zero input duties and returns it
without ordinary target results. Close that lifecycle gap in one common finalizer.

1. Construct the optimized case using the winning canonical utility definitions.
2. Allocate on its exact frozen residual, or use the existing direct/aggregate
   targeting route selected by ordinary placement. This is a final deterministic
   replay, not another optimization or HPR solve.
3. Populate the result cache, selected-period results and graphs for exactly the
   requested zone and periods. Do not use a blanket all-period/all-zone call that
   expands the study beyond the placement request.
4. Attach/retain utility_placement_result after finalization, accounting for the
   current transaction snapshot's clearing of that field. Both the final thermal
   results and placement evidence must survive.
5. Verify allocated duties, capacity bounds, utility identity and feasibility
   agree with the winning candidate within existing numerical tolerances.
6. Return only after finalization succeeds. Summary, plot, report and export
   access must read cached results without launching an analysis.

Input JSON remains configuration, not a serialized result cache. Serializing a
solved optimized case to ordinary input JSON and constructing a new problem from
it still produces an unsolved case; workspace result persistence uses its own
existing contract. Document that distinction explicitly.
