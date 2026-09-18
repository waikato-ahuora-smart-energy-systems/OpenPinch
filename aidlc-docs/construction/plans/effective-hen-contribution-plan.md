# Effective HEN temperature contribution correction

## Requirements and scope
Use the prepared stream effective_delta_t_contribution, in delta_degC, for each
operating period. Preserve positive values above tol directly, including values
below dTmin / 2. Use dTmin / 2 only when the effective value is at or below tol.
Apply the same adapter rule to process streams, utilities and their segments.
The latest user instructions authorize this focused correction and supersede
the working-tree max floor. The follow-up extends the same effective-value/fallback policy to PDM targeting.

## Adaptive workflow
Existing Python library and reverse-engineering artifacts reused. Minimal
requirements and one code unit; stories, architecture, units, NFR and
infrastructure stages skipped because this is an isolated numerical correction.
No operations or deployment work. Existing extension choices retained:
Property-Based Testing enabled; Security and Resiliency disabled.

## Testable properties
Effective contributions above tolerance are invariant under changes in solver
dTmin. Effective values include the domain multiplier exactly once. Period and
segment identity are preserved, inputs remain unchanged, and zero falls back.
Examples cover below/above fallback and tol boundaries; generated segmented
streams cover varied shapes and multipliers with fixed-seed Hypothesis.
PBT-01/03/05/07/08/09/10 apply; PBT-02 N/A (no inverse), PBT-04 N/A (no
normalization operation), PBT-06 N/A (adapter is read-only).

## Execution
- [x] 1. Inspect workspace, requirements, callers and existing tests.
- [x] 2. Add regressions for effective contributions and fallback-only semantics.
- [x] 3. Verify the user edit in arrays.py reads effective contributions without scaling or flooring.
- [x] 4. Run affected HEN tests and lint; record results and remaining policy difference.

## PDM follow-up requirements and design
User explicitly requests PDM preserve original contributions and use dTmin / 2
only for missing or near-zero values. Effective contributions remain authoritative
as established earlier. Capture all effective stream and segment values before
normalizing copied multipliers; then materialize those captured values in the
copied targeting zone. Never mutate the original PinchProblem. Use the same
scalar helper as solver arrays to keep tolerance/fallback behavior identical.
Update decomposition metadata to describe fallback rather than a floor.

Testable properties: copied effective values equal solver-array contributions;
positive effective values are independent of dTmin; repeated normalization is
idempotent; original values, multipliers and locks remain unchanged. Examples
cover near-zero boundaries, missing values, multiperiod segments and utilities.
PBT-04 now applies to normalization and will be checked on generated segments.

- [x] 5. Inspect PDM callers and define the follow-up correction.
- [x] 6. Add failing regressions for copied-zone effective values and fallback.
- [x] 7. Implement shared contribution semantics and update metadata.
- [x] 8. Run affected tests and HEN checks; update verification records.

## PR 98 metadata correction

User approved the proposed metadata-only correction with Go. Restore the
active fallback input and synthesis multiplier setting, remove the misleading
active contribution multiplier, and describe effective-value fallback semantics.
No numerical logic changes; existing PBT coverage remains applicable and unchanged.

- [x] 9. Inspect current diff and confirm the approved metadata scope.
- [x] 10. Correct metadata and verify serialized output, adapter tests and lint.
