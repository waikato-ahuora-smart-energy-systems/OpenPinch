# Unit 1 NFR Requirements

## Scope

These requirements govern the HPR performance-map contract, its JSON Schema,
golden fixtures, validators, and package resources. Thermodynamic engine
execution and public application integration belong to Units 2 and 3.

## Compatibility and stability

### NFR-U1-001: Supported runtime

The contract must run on the repository's supported Python `>=3.14.2` baseline
and Pydantic major version 2 (`pydantic<3`). It must not require a second model
framework or consumer package.

**Verification**: focused tests run in the repository environment and the
installed-wheel smoke imports and validates both fixtures.

### NFR-U1-002: Closed alpha interchange

Only schema version `1.0` is accepted. The published field set, enum values,
units, defaults, and physical invariants must match the corrected OpenUtility
alpha consumer. Contract changes that alter accepted data require an explicit
schema-version decision; they may not silently broaden `1.0`.

**Verification**: schema snapshot, unknown-version, extra-field, exact-unit,
and mode/convention tests plus external fixture-consumer evidence.

### NFR-U1-003: Existing API compatibility

Adding the specialist contract module and package data must not change existing
`OpenPinch` root exports, current HPR target return types, or existing contract
serialization. Unit 1 exposes no new root workflow.

**Verification**: root inventory and complete existing contract regression
tests remain unchanged and green.

## Performance and scalability

### NFR-U1-004: Linear validation

Validation time and additional working memory must scale linearly with the
number of points plus nested provenance values. Validation must group and check
curves in one bounded pass and must not perform pairwise point comparison.

**Verification**: design/code review confirms `O(points + provenance_values)`
behavior; a focused generated-map performance test validates 10,000 points in
at most 2.0 seconds on the supported project test environment.

### NFR-U1-005: No hidden expensive work

Importing the contract module or loading its type definitions must not read
fixtures, generate JSON Schema, traverse performance points, import a
thermodynamic engine, or access the network/filesystem outside Python's normal
module loading.

**Verification**: cold-import architecture test with blocked engine imports;
resource generation is invoked only by an explicit development command/test.

### NFR-U1-006: Bounded resource artifacts

The two golden fixtures and one JSON Schema are static package resources. Each
golden fixture contains the minimum representative curve set needed for
consumer verification and remains below 100 KiB; the schema remains below
250 KiB.

**Verification**: package-resource size assertions on source and built wheel.

## Reliability and determinism

### NFR-U1-007: Fail-closed atomic construction

An invalid request or map returns no contract value or partial fixture. All
field, point, curve, physical, and provenance checks execute before the map is
made available to Units 2 or 3.

**Verification**: example and generated invalid payloads assert failure and
absence of partial output.

### NFR-U1-008: Deterministic canonical output

Equivalent canonical contract inputs must produce equal model values, the same
point order, structurally identical mappings, and byte-identical JSON text,
schema, and fixture resources on every supported run.

**Verification**: repeated-generation byte comparisons and fixed-seed
Hypothesis properties.

### NFR-U1-009: Actionable stable diagnostics

Every validation failure must include a precise field/element location and a
stable invariant-focused message fragment. Tests may depend on category and
message fragment but not a full implementation-specific stack or complete
Pydantic rendering.

**Verification**: table-driven failures cover structure/type, version/unit,
range/finiteness, mode convention, physical point, curve topology, and
provenance categories.

## Dependency and environment isolation

### NFR-U1-010: Engine-independent cold imports

The contract module, schema, fixtures, and resource helpers must import when
TESPy is absent or its import is actively blocked. Unit 1 must not import
CoolProp, TESPy, OpenUtility, Pyomo, HiGHS, or HPR runtime target objects.

**Verification**: subprocess cold-import with TESPy and external optimizer
packages blocked, plus a static test forbidding any direct CoolProp, TESPy,
runtime HPR, or optimizer import in Unit 1.

### NFR-U1-011: No new runtime dependency

Unit 1 must use existing runtime Pydantic and Python standard-library JSON and
resource facilities. Independent JSON Schema conformance may add `jsonschema`
only to the development dependency group, not the runtime or optional HPR
profiles.

**Verification**: project metadata and installed-wheel dependency inspection.

### NFR-U1-012: Offline operation

Contract validation, serialization, schema generation, fixture regeneration,
and tests must require no network, service account, external process, solver,
or system executable.

**Verification**: focused suite passes under the normal offline test sandbox.

## Data safety

### NFR-U1-013: Data-only deserialization

External maps are treated as untrusted JSON-like data. Unit 1 must not use
pickle, `eval`, dynamic import names, object hooks that instantiate arbitrary
types, or lossy string coercion. Only the declared recursive JSON value domain
is accepted.

**Verification**: forbidden-API architecture test and hostile-value examples
for bytes, paths, sets, object instances, non-string keys, NaN, and infinity.

### NFR-U1-014: Detached serialization

Serialized mappings/lists must not share mutable state with immutable contract
values. Mutating a returned plain-data payload must not change the source map or
a later serialization.

**Verification**: example and generated detachment tests.

Authentication, authorization, confidentiality, uptime, disaster recovery,
and failover requirements are N/A because Unit 1 is an offline in-process data
contract with no persistence, network listener, credentials, or production
service state.

## Maintainability and testability

### NFR-U1-015: One authoritative model

Field definitions and semantic validation must have one owner in
`OpenPinch.contracts.hpr_performance_map`. JSON Schema and fixtures derive from
that owner. Documentation and consumer examples must not contain a second
hand-maintained Python contract.

**Verification**: generation/drift test and static ownership review.

### NFR-U1-016: Complete rule evidence

BR-001 through BR-027 must each map to at least one example test, property test,
or resource verification. Critical heat-pump and refrigeration fixture behavior
must have both example-based and property-based coverage.

**Verification**: Code Generation test matrix and focused branch coverage of at
least 95 percent for the new contract/resource helpers.

### NFR-U1-017: Reproducible property testing

Hypothesis must use reusable domain strategies, retain automatic shrinking, run
in CI, and use or report a replayable seed under the repository's established
policy. A shrunk defect becomes a permanent example regression.

**Verification**: PBT configuration, fixed seed/reporting evidence, and test
organization review.

### NFR-U1-018: Documentation usability

The specialist module and resources must document every map and point field,
mode-specific formula, tolerance meaning, alpha status, unsupported zero-load
point policy, and plain-JSON consumer path. Validation errors must be usable
without knowledge of thermodynamic engine internals.

**Verification**: warning-strict documentation build and installed-package
resource example.

## Packaging

### NFR-U1-019: Distribution completeness

JSON Schema and both golden fixtures must be present and byte-identical in the
source tree, source distribution, and wheel. Resource access must use package
resource APIs and work without assuming an unpacked checkout.

**Verification**: sdist/wheel inventory, isolated installation, and resource
read/validation smoke.

### NFR-U1-020: Patch and quality hygiene

Unit 1 changes must pass Ruff lint and formatting, focused pytest and
Hypothesis tests, warning-strict documentation, build, package validation,
dependency audit, installed-wheel smoke, and `git diff --check`.

**Verification**: Unit 1 and integrated Build and Test evidence.

## Extension compliance

Property-Based Testing is enabled:

- PBT-09 is satisfied by the existing Hypothesis/pytest stack.
- PBT-08 requires shrinking and reproducible seed evidence.
- PBT-02, PBT-03, PBT-05, PBT-07, and PBT-10 requirements are measurable in
  NFR-U1-008, NFR-U1-014, NFR-U1-016, and NFR-U1-017.
- PBT-04 and PBT-06 remain N/A for immutable, stateless contract values.

Security Baseline and Resiliency Baseline are disabled in project state. Their
extension-specific requirements are not enforced; ordinary data-safety and
reliability requirements above remain part of Unit 1.
