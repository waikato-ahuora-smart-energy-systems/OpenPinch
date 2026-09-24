# Unit 3 Business Logic Model

## Direct process-MVR algorithm

1. Validate stream, period, stage count, efficiencies, and exactly one
   compression target.
2. Reject non-finite, non-positive, or out-of-domain per-stage requests before
   stage iteration: lift greater than 200 degC or pressure ratio greater than 20.
3. Resolve required inlet, compression, target, and profile states. Translate a
   property `ValueError` into one bounded contextual domain error naming source
   stream, period, stage, and fluid, with the original exception chained.
4. Permit liquid-injection saturation failure only as `dry_stage`, and permit
   optional saturation-breakpoint failure only as `reduced_profile`.
5. Attach immutable fallback diagnostics to the owning stage result.
6. Return replacement streams only after every requested stage succeeds.

## Properties

| ID | Category | Property |
|---|---|---|
| U3-P1 | Invariant | Supported lift and ratio values are finite and inside closed documented bounds |
| U3-P2 | Boundary | Values at each upper bound are accepted and values above are rejected before stage solving |
| U3-P3 | Invariant | Required-state errors contain stream, period, stage, fluid, and stable reason code |
| U3-P4 | Invariant | Required-state errors retain a causal exception and no raw engine message in public text |
| U3-P5 | Model property | A required-stage failure returns no partial replacement result |
| U3-P6 | Invariant | Fallback codes belong to exactly `dry_stage` or `reduced_profile` |
| U3-P7 | Invariant | Each fallback diagnostic is detached, immutable, bounded, and stage-specific |
| U3-P8 | Metamorphic | Enabling injection with unavailable optional saturation equals dry-stage thermodynamics plus evidence |
| U3-P9 | Metamorphic | Unavailable optional profile breakpoints preserve endpoint duty plus evidence |
| U3-P10 | Regression | Packaged normal direct process-MVR remains numerically compatible |
| U3-P11 | Integration | Accepted optimized targets remain detached and copy-safe |
| U3-P12 | Executability | Notebooks distinguish physical typed failures from unexpected service defects |
