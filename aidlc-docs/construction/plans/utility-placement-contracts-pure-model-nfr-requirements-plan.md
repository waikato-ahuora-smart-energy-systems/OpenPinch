# Utility Placement Contracts and Pure Model NFR Requirements Plan

## Unit Assessment

Unit 1 is an in-process, deterministic contract and pure-transformation module.
Its scale drivers are utility levels, decision coordinates, periods, nested
result size, and generated test examples. It has no independent uptime,
network, persistence, authentication, authorization, disaster-recovery, UI, or
accessibility boundary. Existing repository quality, units, packaging, and
optional-dependency constraints remain authoritative.

## Plan Steps

- [x] Analyze the approved Unit 1 Functional Design, 43 business rules, domain
  entities, PBT-01 property inventory, and repository quality configuration.
- [x] Collect and analyze every NFR answer below; add follow-up questions for
  any vague, combined, missing, or contradictory response.
- [x] Obtain explicit approval of the answered NFR Requirements plan.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-requirements/nfr-requirements.md`.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-requirements/tech-stack-decisions.md`.
- [x] Verify PBT-09 framework selection, dependency presence, shrinking,
  fixed-seed reproducibility, and pytest integration.
- [x] Validate measurable NFR coverage, Functional Design consistency, and
  disabled Security/Resiliency extension handling.
- [x] Obtain explicit approval before Unit 1 NFR Design.

## Question 1 - Scalability and Capacity

What capacity policy should Unit 1 expose for utility counts and periods?

A) Impose no arbitrary public maximum in the first release; require memory/time
complexity linear in period-coordinate input size, document representative
tested capacities, and rely on Unit 2's bounded optimiser options for expensive
search control

B) Add hard public maxima for level counts and periods and describe the exact
limits after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 2 - Performance Benchmark

What measurable pure-model performance requirement should apply?

A) Require request/template normalization, envelope intersection, ordering
propagation, vector-schema construction, and primary-start verification to
complete together within 250 milliseconds on the project CI runner for 20
levels per side across 100 periods, with no worse than linear growth in the
number of period-coordinate bounds

B) Use a different benchmark size or threshold and state it after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 3 - Numerical Tolerance Defaults

Which default comparison policy should the contracts and pure model use?

A) Reuse OpenPinch's `tol = 1e-6` for absolute, bound, and ordering comparisons;
use `1e-9` relative tolerance for scaled float comparison; normalize equality
within bound tolerance to one fixed coordinate; and allow validated explicit
overrides

B) Use different named defaults and state their exact values after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 4 - Availability and Continuity

What availability requirement applies to this in-process pure module?

A) Mark uptime, failover, disaster recovery, and business continuity as N/A;
require deterministic return or typed failure with no partial state because the
module owns no service process or persistent state

B) Introduce an independent availability/recovery boundary and describe it
after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 5 - Security and Data Protection

What security scope should Unit 1 implement while the Security extension is
disabled?

A) Add no authentication, authorization, encryption, credential, network, or
file handling; enforce strict extra-forbid schemas, finite/size-consistent
collections, safe serializable values, and no retention of caller/runtime
objects as ordinary library validation

B) Enable additional security/compliance requirements and describe the threat
model after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 6 - Technology Stack and PBT-09

Which implementation and test stack should Unit 1 use?

A) Use Python 3.14.2+, Pydantic v2 contracts, existing OpenPinch/Pint value and
unit conversion, standard-library immutable structures and math, pytest, and
Hypothesis with existing fixed seed `20260715`; add no mandatory runtime or test
dependency

B) Choose a different contract, unit, or property-testing stack and identify
the exact replacement after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`; Python minimum corrected
to the repository's existing `>=3.14.2` contract)

## Question 7 - Reliability and Maintainability Gates

Which delivery gates should be required for Unit 1?

A) Require focused examples plus all PBT-01 properties, fixed-seed shrinking,
permanent regressions for shrunk failures, Ruff, architecture/import checks,
branch coverage compatible with the repository's 95 percent gate, full
non-solver regressions, typed public signatures, docstrings, and no broad
exception suppression

B) Use a different gate set and describe exact additions/removals after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 8 - Usability and Compatibility

What interface-quality requirement should apply to specialist contracts?

A) Preserve the two-symbol package-root facade; expose types from their concrete
specialist module; keep stable snake-case fields and enum values; provide
field/template/period-aware errors; preserve collection order and explicit
units in JSON; and document any future breaking schema change under normal
package compatibility policy

B) Expand root exports or use another compatibility policy and describe it
after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)
