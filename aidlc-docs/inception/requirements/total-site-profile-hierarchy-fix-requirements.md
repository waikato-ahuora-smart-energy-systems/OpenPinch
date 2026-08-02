# Total Site Profile Hierarchy Fix Requirements

## Intent Analysis

- **Request type**: Bounded brownfield defect correction.
- **Goal**: Build each Total Site process composite curve and its paired utility
  profile from the same immediate-subzone Direct Integration target layer.
- **Complexity**: Moderate state-ownership correction across direct, indirect,
  and multi-period targeting; no new public capability.
- **Primary evidence**: Notebook 2 overstates both process composite-curve duties
  by 59,168.043 kW because Site targeting consumes Unit Operation profiles while
  utility aggregation consumes Process Zone targets.

## Functional Requirements

1. Total Site targeting shall reconstruct immediate-subzone net hot and cold
   segments from each current Direct Integration target's `H(net)-actual` GCC,
   targeted utilities, and selected period.
2. Every zone shall own two explicit hot/cold net-profile pairs:
   `net_hot_streams` / `net_cold_streams` for its own Direct Integration GCC,
   and `subzone_net_hot_streams` / `subzone_net_cold_streams` for profiles
   reconstructed from its immediate subzones' Direct Integration targets.
3. Reconstructed immediate-subzone segments shall be combined in fresh,
   period-aware collections with stable subzone-qualified keys and assigned to
   the second pair before Total Site targeting consumes them.
4. A zone's direct `net_hot_streams` and `net_cold_streams` shall not be replaced
   by an indirect aggregation of child profiles.
5. Zones without children shall retain the existing `zone.net_*` fallback used
   by low-level callers.
6. Normal and multi-period indirect targeting shall use the same deterministic
   reconstruction behavior.
7. Graph labels, graph schemas, target APIs, notebook calls, and final Total Site
   utility targets shall remain compatible.

## Acceptance Criteria

- Notebook 2 Cold CC duty is approximately 212,431.388 kW and matches summed
  immediate-subzone Hot Utility duty within `rtol=1e-6`, `abs=0.2 kW`.
- Notebook 2 Hot CC duty is approximately 115,316.151 kW and matches summed
  immediate-subzone Cold Utility duty within the same tolerance.
- Final Total Site targets remain approximately 180,094.613 kW hot utility and
  82,979.376 kW cold utility.
- Repeated `all_heat_integration()` and focused Total Site targeting are
  idempotent for the plotted profiles.
- Poisoned mutable child `zone.net_*` state cannot affect parent Total Site
  profiles when valid child Direct Integration targets exist.
- Notebook 2 SUGCC rendering shall preserve the constant-enthalpy HPS-to-LPS
  connection and the approximately 138.5 degC LPS ledge after graph rounding.
- Consecutive duplicate graph coordinates shall not remove an internal corner
  or merge distinct utility segments.

## Non-Functional Requirements

- Add only the requested Zone profile properties; do not add dependencies,
  target schemas, target fields, or deployment configuration.
- Preserve existing notebook improvements and unrelated working-tree files.
- Keep reconstruction deterministic across periods and repeated executions.

## Extension Configuration

- Security Baseline: Disabled; no security boundary changes.
- Resiliency Baseline: Disabled; no operational system changes.
- Property-Based Testing: Partial. PBT-03, PBT-07, PBT-08, and PBT-09 apply to
  the reconstruction invariant; PBT-02 is N/A because there is no round trip.

## Approval

The user's complete implementation plan supplies the requirements, acceptance
criteria, technical constraints, and explicit authorization to implement them.
