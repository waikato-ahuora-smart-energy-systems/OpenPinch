# Maximum-Duty Business Rules

1. `maximum_duties` is optional and keyed by final globally unique utility
   names; it is not an optimizer option.
2. Limits accept existing scalar, scalar-with-unit, and period-resolved value
   forms and normalize to the configured heat-flow unit.
3. Limits are finite and non-negative; unknown names, invalid units, and
   incomplete selected periods fail before optimizer execution.
4. Omitted names are unbounded and zero limits disable only that named level.
5. Every named duty satisfies `Q <= Q_max` independently in every period.
6. Generated hot/cold temperature pairs share temperatures but never share a
   duty cap.
7. The cascade exhausts eligible named capacity in normal temperature order
   before assigning residual fallback duty.
8. Default `HU` and `CU` are not counted placement levels or decision
   coordinates and never displace available named capacity.
9. Positive fallback duty remains feasible, participates in the physical
   balanced-composite entropy calculation, and is visible in result evidence.
10. `g_penalty()` is the squared normalized fallback fraction and aggregates
    with raw period weights; it is never labelled as entropy.
11. Returned cases retain limit metadata and any used fallback so normal
    retargeting and standard plots reproduce capacity-constrained allocation.
12. The source problem, source utilities, cached targets, and active workspace
    case remain unchanged.
