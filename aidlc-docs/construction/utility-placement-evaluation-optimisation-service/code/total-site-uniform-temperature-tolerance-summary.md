# Total Site Uniform Temperature Tolerance Summary

## Outcome

Total Site utility placement no longer fails when two candidate utility
endpoints differ by temperature noise at the shared domain tolerance. Process
and site utility profiles now make the same decision about whether such
breakpoints represent one temperature interval.

## Root cause

Temperature grids previously used three different equivalence rules. Cascade
construction rounded and deduplicated values, interval insertion used an
absolute tolerance, and `ProblemTable.update` used both an absolute and a
scale-dependent relative tolerance. A candidate could consequently produce a
70-row utility profile and a 71-row Total Site table before cumulative utility
columns were assigned.

## Correction

One internal temperature-grid implementation now owns:

1. finite-value filtering and rounding to the domain resolution;
2. descending near-duplicate removal;
3. missing-interval and interior-interval decisions;
4. interpolation denominator safeguards; and
5. equal-grid checks before problem-table updates.

The comparison remains absolute. Numerical normalization is performed six
decimal places below the `1e-6 K` domain tolerance solely to remove binary
subtraction noise at the tolerance boundary. No private zero-tolerance or
relative-tolerance alignment path remains.

## Verification

The fixed saved failure, a negative-temperature boundary case, an explicit
high-temperature absolute-only check, and 25 generated valid adjacent-utility
profiles verify grid size, descending order, finite cumulative values, and duty
conservation. The captured four-isothermal Total Site regression solved and its
detached result completed ordinary Total Site retargeting with 71 rows.
