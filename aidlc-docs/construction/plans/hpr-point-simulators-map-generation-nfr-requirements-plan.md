# Unit 2 HPR Point Simulators and Map Generation NFR Requirements Plan

## Scope

Define measurable compatibility, dependency-isolation, performance,
reliability, numerical-reproducibility, maintainability, and testing
requirements for the default CoolProp simulator, optional TESPy leaf, and
deterministic map-generation service. Functional behavior remains governed by
the approved Unit 2 Functional Design.

## Assessment Plan

- [x] Read the approved Functional Design, Unit 2 definition, dependency
  boundaries, Unit 1 contract NFRs, package metadata, CI seed policy, and
  enabled Property-Based Testing rules.
- [x] Evaluate scalability, performance, availability, security, technology,
  reliability, maintainability, and usability applicability for this local
  Python-library unit.
- [x] Identify material NFR decisions for optional dependency packaging,
  supported TESPy versions, real-engine CI, floating-point reproducibility,
  mixture property backends, concurrency, and grid-performance policy.
- [x] Create complete mutually exclusive questions with answer tags.
- [x] Validate every user answer and resolve ambiguity or contradiction.
- [x] Define measurable compatibility and optional-dependency requirements.
- [x] Define deterministic numerical, tolerance, provenance, and mixture
  requirements across supported environments.
- [x] Define linear orchestration, resource cleanup, failure isolation, and
  concurrency requirements.
- [x] Define example, real-engine, property-based, architecture, packaging, and
  quality-gate verification requirements.
- [x] Document technology-stack decisions, rejected alternatives, and complete
  NFR/extension traceability.
- [x] Validate the NFR artifacts and update plan/stage/audit tracking.
- [x] Request explicit NFR Requirements approval before Unit 2 NFR Design.

## Applicability Assessment

- Scalability applies to Cartesian grid expansion, diagnostic aggregation, and
  per-call engine session resources; horizontal service scaling is N/A.
- Performance applies to orchestration overhead and bounded regression tests;
  engine solve time is environment and fluid dependent.
- Availability, disaster recovery, authentication, authorization, persistent
  storage, and accessibility are N/A because this unit is an in-process
  library with no service, account, database, or UI.
- Ordinary safe-data handling, bounded diagnostics, dependency isolation, and
  cleanup remain required even though the Security and Resiliency extensions
  are disabled.
- PBT-09 applies and requires the existing Hypothesis/pytest selection to be
  documented. PBT-08 and the properties identified under PBT-01 inform the
  measurable test requirements carried into Code Generation.

## NFR Decision Questions

Please fill every `[Answer]:` tag with one listed letter. Choose `X` and add a
description when none of the listed options matches the intended requirement.

### Question 1
How should users install TESPy support now that it serves both Brayton and HPR
features?

A) Add a neutral `tespy` optional extra, retain `brayton_cycle` as a compatible
alias, and keep TESPy in `full`. This is the recommended discoverable path.

B) Add a dedicated `hpr_tespy` extra while retaining the existing
`brayton_cycle` extra separately.

C) Reuse only the existing `brayton_cycle` extra for HPR TESPy support.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A


### Question 2
What TESPy version-compatibility policy should the first HPR adapter use?

A) Declare a tested minimum of `tespy>=0.10.1.post2` without an upper bound,
and let blocking compatibility tests detect future breakage. This is the
recommended library-friendly policy.

B) Restrict the first release to `tespy>=0.10.1.post2,<0.11` until a later
compatibility review deliberately broadens the range.

C) Keep an unversioned `tespy` dependency and rely only on the lockfile used by
development and CI.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 3
How should real TESPy behavior be enforced in continuous integration?

A) Add a blocking optional-profile job with real pure-fluid, registered-blend,
explicit-mixture, and design/offdesign smoke tests, while a separate base
profile proves imports and CoolProp behavior without TESPy. This is the
recommended dependency-boundary evidence.

B) Run real TESPy smoke tests only when TESPy happens to be installed in the
ordinary full development job; skip them otherwise.

C) Keep real TESPy tests local and non-blocking; CI uses only fake simulators.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 4
What numerical reproducibility contract should apply across platforms and
engine patch versions?

A) Require exact structure, ordering, identities, and provenance, but compare
thermodynamic floating-point results with declared physical tolerances. Do not
quantize valid engine outputs. This is the recommended scientifically honest
contract.

B) Round every generated duty, power, and COP value to a fixed decimal count so
serialized maps are byte-identical across supported environments.

C) Require bit-for-bit identical thermodynamic values and fail compatibility
tests on any floating-point difference.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 5
Which property backends should the mixture contract permit through CoolProp and
TESPy?

A) Use HEOS by default and preserve an explicitly requested backend such as
REFPROP when it is installed and licensed; required CI uses HEOS and does not
depend on REFPROP. This is the recommended portable policy.

B) Support HEOS only in the first release and reject every explicit alternative
property backend.

C) Require REFPROP for every explicit refrigerant mixture and use HEOS only for
pure or registered fluids.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: REFPROP should be rejected; only CoolProp (>=8) should be used.

### Question 6
What concurrency guarantee should map generation provide?

A) Use a fresh, unshared session and temporary state per call, avoid OpenPinch
global mutable state, and document that external engine thread safety is not
guaranteed; callers needing parallel generation use isolated processes. This is
the recommended conservative guarantee.

B) Add a process-global lock that serializes every TESPy generation call while
allowing CoolProp calls to run concurrently.

C) Reject overlapping generation calls explicitly for both backends.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 7
What production grid-size and performance policy should the generator enforce?

A) Impose no arbitrary production point cap or wall-clock timeout; require
linear orchestration and diagnostic memory, a bounded 10,000-point fake-engine
benchmark, and functional rather than timing assertions for real TESPy. This is
the recommended engine-independent policy.

B) Add configurable maximum-point and per-map timeout settings to Unit 2.

C) Reject every request above a fixed 1,000-point limit.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Content Validation

This plan contains no Mermaid diagram, ASCII diagram, executable code block,
embedded JSON, or YAML. Markdown headings, lists, option spacing, answer tags,
paths, and checkbox syntax were validated before creation.
