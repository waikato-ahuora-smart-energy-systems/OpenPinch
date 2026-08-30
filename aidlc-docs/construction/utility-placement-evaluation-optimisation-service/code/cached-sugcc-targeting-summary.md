# Cached Process-Profile and SUGCC Targeting Summary

## Outcome

Aggregate utility-placement candidates no longer rebuild a problem tree or
rerun Direct and Indirect Integration targeting. Each selected process and
period owns one completed, utility-independent net-load profile. Candidate
evaluation inserts only the candidate utility endpoints into copies of those
profiles, calculates duties with the canonical targeting algorithm, and builds
one candidate SUGCC from the accumulated duties.

## Calculation boundary

The cached aggregate path performs the following work once:

1. prepare the process problem table and zonal targets;
2. add constant-heat breakpoints, pockets, assisted-GCC data, and separated
   hot and cold load profiles; and
3. reconstruct the invariant aggregate process composite used by balanced-
   composite entropy evaluation.

For each candidate it then:

1. creates candidate utility Streams directly from the decoded placement;
2. interpolates candidate utility temperatures into each cached child profile;
3. calculates child duties without mutating the reusable candidate Streams;
4. sums duties by utility level and constructs one SUGCC through the existing
   site-utility profile owner; and
5. applies the existing same-temperature generation/use cancellation before
   returning named duties, fallbacks, targets, and the entropy snapshot.

Direct-scope placement retains prepared ordinary replay. Full hierarchy replay
remains available as the independent test oracle and ordinary targeting of the
returned case remains the public acceptance workflow.

## Correctness evidence

Five structured four-level candidates reproduce full hierarchy replay for
every named duty, fallback duty, and hot/cold target. Their thermodynamic
objectives agree within `1e-4 kW/K`, including the cached invariant process
composite. Fixed and generated child-profile tests reproduce ordinary utility
duties; lifecycle tests prove profiles are completed once; hierarchy,
multiperiod, pickle, repeated-replay, and source-isolation regressions pass.

## Performance evidence

Ten representative Total Site candidates take `0.294159917 s` through the
cached path versus `3.263975833 s` through full hierarchy replay. This is an
`11.0959` times speed-up with identical duties and required targets. The
remaining per-candidate work is limited to temperature interpolation, canonical
utility targeting, numeric duty accumulation, one SUGCC, and same-level
cancellation.
