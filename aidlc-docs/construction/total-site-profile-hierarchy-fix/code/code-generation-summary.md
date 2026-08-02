# Total Site Profile Hierarchy Fix Implementation Summary

## Outcome

Total Site process composite curves now use fresh net profiles reconstructed
from each immediate subzone's current Direct Integration target. The paired
utility profiles already use those same targets, so both sides of the Total
Site calculation now share one hierarchy level and one selected period. Both
profile ownership levels are persistent and explicit on every `Zone`.

## Implementation

- Added an internal reconstruction helper in Total Site targeting. It rebuilds
  net hot and cold segments from each child target's `H(net)-actual` cascade,
  targeted utilities, and target period.
- Retained `net_hot_streams` and `net_cold_streams` exclusively for the present
  zone's Direct Integration GCC profiles.
- Added `subzone_net_hot_streams` and `subzone_net_cold_streams` for profiles
  reconstructed from immediate-subzone Direct Integration targets, plus the
  combined `subzone_net_process_streams` view.
- Added stable child-qualified stream keys and period context to freshly
  populated second-pair collections. Mutable child `net_*` collections are not
  read during deterministic reconstruction.
- Routed explicit net-profile imports into the second pair, preserving the
  present zone's own Direct Integration pair.
- Retained the existing `zone.net_*` fallback when a zone has no children.
- Removed child-profile imports from normal indirect targeting and multi-period
  indirect HPR preparation. A zone's `net_*` state therefore remains its own
  direct-GCC-derived profile.
- Preserved all existing graph serialization, labels, APIs, schemas, target
  values, and notebook calls while adding the requested Zone properties.
- Corrected composite-curve cleanup to remove consecutive identical coordinate
  pairs before collinearity analysis. Rounded SUGCC data therefore retains
  internal constant-enthalpy corners instead of joining adjacent utility levels
  diagonally.

## Regression Evidence

- `pulp_mill.json` Cold CC is 212,431.388 kW instead of 271,599.431 kW.
- `pulp_mill.json` Hot CC is 115,316.151 kW instead of 174,484.194 kW.
- The shared 59,168.043 kW overstatement is removed.
- Final Total Site targets remain 180,094.613 kW hot utility and 82,979.376 kW
  cold utility within the accepted numerical tolerance.
- Poisoned child net-profile state, repeat targeting, direct-profile ownership,
  empty profiles, no-child fallback, selected periods, and multi-period HPR
  preparation are covered.
- Domain tests prove the two pairs are distinct, receive period context, and
  coexist at Site, Process Zone, and Unit Operation levels.
- Notebook 2 graph tests prove HPS ends at approximately 27,253.71 kW and 279
  degC, connects vertically to 138.5254 degC, and retains the LPS ledge to
  approximately 11,259.17 kW and 138.4254 degC.

## Extension Compliance

| Extension rule | Status | Evidence |
|---|---|---|
| Security Baseline | N/A | Disabled; no security boundary changed. |
| Resiliency Baseline | N/A | Disabled; no operational service changed. |
| PBT-02 round trip | N/A | Reconstruction has no inverse operation. |
| PBT-03 invariant | Compliant | Duty conservation and poisoned-state independence are properties. |
| PBT-07 input strategy | Compliant | Structured bounded subzone duties generate one to five profiles. |
| PBT-08 shrinking/replay | Compliant | Hypothesis shrinking retained with seed `20260715`. |
| PBT-09 framework | Compliant | Existing Hypothesis and pytest stack used. |
