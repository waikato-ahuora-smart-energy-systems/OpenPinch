# 5. Utility transfer semantics

Proposed signature: problem.with_utilities_from(other_problem, *, project_name=None).
The donor is a prepared PinchProblem, whether solved or unsolved. Copy canonical
utility definitions rather than reverse-engineering them from duty summaries or
placement-result reports. Preserve the receiver's project name unless overridden.

- Preserve receiver streams, network/profile input where applicable, zone tree,
  period configuration, options and any existing receiver residual basis.
- Replace the utility list as a whole. Copy temperatures, units, nominal segment
  proportions, profiles, fluid metadata, active status, costs, heat-transfer
  coefficients, approach contributions and maximum-duty constraints.
- Keep nominal segment duties that define shape; do not confuse those with
  allocated operating duties. Do not flatten or erase a segmented profile.
- Never copy the donor's residual basis, process streams, HPR result cache,
  placement result or analysis-owner identity into a different receiver.
- Normalize explicit period arrays by period identity, not position. Initially
  require matching receiver/donor period sets for this convenience operation;
  normalize order and reject unsupported subset/broadcast mapping with a clear
  message. An explicit mapping API is a separate extension.
- Validate the replacement input through the existing canonical schemas. Reuse
  component-aware cloning/preparation when the receiver has runtime components
  that ordinary input JSON cannot faithfully represent; never silently drop
  an active component. Test receiver component membership and independent owners.
- Return a fresh unsolved problem. Subsequent input edits affect only that new
  case. The new case is not claimed to retain the donor's optimized objective
  when the receiver has a different thermal background.

The short JSON example discussed earlier is the simple-case equivalence oracle;
the convenience implementation must also handle the existing richer case state
without silently losing information.
