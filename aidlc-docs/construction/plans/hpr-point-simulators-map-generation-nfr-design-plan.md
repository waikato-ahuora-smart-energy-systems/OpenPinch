# Unit 2 HPR Point Simulators and Map Generation NFR Design Plan

## Scope

Translate the approved Unit 2 Functional Design and NFR Requirements into
concrete resilience, scalability, performance, security, module-boundary,
resource, convergence, CI-profile, and testability patterns. Infrastructure
services remain out of scope because Unit 2 is an in-process Python library.

## Design Plan

- [x] Read the approved Unit 2 Functional Design, all 28 NFR requirements,
  technology decisions, dependency graph, and enabled Property-Based Testing
  rules.
- [x] Evaluate resilience, scalability, performance, security, and logical
  component categories for applicability.
- [x] Identify unresolved pattern choices for failure classification, lazy grid
  traversal, per-session caching, sanitized diagnostics, module layout,
  deterministic convergence settings, and characteristic resources.
- [x] Create complete mutually exclusive questions with answer tags.
- [x] Validate every user answer and resolve ambiguity or contradiction.
- [x] Define NFR design patterns with complete NFR-U2 traceability.
- [x] Define logical components, allowed dependencies, lifecycle ownership, and
  public/private boundaries.
- [x] Define base/TESPy CI profile composition, PBT state model, real-engine
  smokes, performance guard, and dependency-upgrade gates.
- [x] Validate content, identifiers, NFR coverage, extension compliance, and
  patch hygiene.
- [x] Update plan, stage, and audit tracking and request explicit NFR Design
  approval before Unit 2 Code Generation planning.

## Category Applicability

- Resilience applies to point-failure classification, atomic map failure,
  cleanup, dependency errors, and unusable engine sessions. Distributed
  retries, circuit breakers, failover replicas, and disaster recovery are N/A.
- Scalability applies to linear grid traversal and diagnostic collection, and
  session memory. Horizontal autoscaling and server capacity are N/A.
- Performance applies to single-resolution context preparation, invariant
  session state, lazy point production, and fake-engine regression thresholds.
- Security extensions are disabled, but sanitized diagnostics, no raw engine
  state, no temporary-path disclosure, and no dependency-driven code execution
  remain ordinary design requirements. Authentication and authorization are
  N/A.
- Logical components apply to the analysis package, simulator adapters,
  characteristic resource loader, diagnostics, and CI profiles. Queues,
  databases, network clients, persistent caches, and infrastructure circuit
  breakers are N/A.

## NFR Design Questions

Please fill every `[Answer]:` tag with one listed letter. Choose `X` and add a
description when none of the listed options matches the intended design.

### Question 1
How should the generator distinguish recoverable point failures from a TESPy
session that can no longer evaluate later coordinates?

A) Each adapter classifies a failure as point-local or session-fatal. A
point-local failure is recorded and the prepared design is restored for the
next coordinate; a fatal failure records the current error and deterministic
`session_unavailable` diagnostics for the remainder. No point is retried. This
is the recommended match to the approved lifecycle.

B) Treat every point failure as session-fatal and never attempt a later
coordinate.

C) Rebuild the engine session and retry a failed point once before classifying
the failure.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 2
How should the service traverse large Cartesian grids while still returning one
complete immutable map or aggregate error?

A) Generate `HprOperatingPoint` values lazily in canonical order and retain only
successful Unit 1 points or diagnostics needed for the final result. Do not
materialize a duplicate operating-point collection. This is the recommended
linear-memory pattern.

B) Materialize the complete `HprOperatingPoint` tuple before simulator
preparation for simpler inspection.

C) Add a public chunked or streaming result API that can return partial map
segments.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 3
Where should parsed fluid data and invariant thermodynamic inputs be cached?

A) Resolve them once inside the fresh generation context/session and discard
them on close. Add no process-global cache. This is the recommended isolated
performance pattern.

B) Add a process-global bounded LRU cache shared by CoolProp and TESPy calls.

C) Reconstruct the property state and every invariant input at every operating
point.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 4
How should raw engine exceptions and logs cross the Unit 2 boundary?

A) Return sanitized typed diagnostics with stable bounded details, retain the
original exception only as an in-process chained cause, and emit no automatic
log entry. This is the recommended library-safe behavior.

B) Return sanitized diagnostics but also log the complete engine exception and
temporary state automatically at warning level.

C) Include complete raw engine exception text in every diagnostic so callers
receive maximum detail.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 5
Where should Unit 2 production code live?

A) Add an internal `OpenPinch.analysis.heat_pumps.performance_maps` package with
engine-neutral core modules and an `adapters` subpackage containing separate
CoolProp and TESPy leaves. This is the recommended enforceable dependency
boundary.

B) Add the generator and both adapters directly to the existing `cycles`
package.

C) Put generation beside the Unit 1 models in `OpenPinch.contracts`.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 6
How should TESPy convergence controls be represented in the first release?

A) Use one OpenPinch-owned versioned immutable settings object with fixed
design/offdesign iteration and residual controls, record its identifier and
values in provenance, and expose no caller tuning in schema `1.0`. This is the
recommended reproducible pattern.

B) Add solver/convergence controls to the Unit 1 public map request so each
caller can tune TESPy.

C) Use the installed TESPy defaults without recording their resolved values.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

### Question 7
How should the OpenPinch-owned compressor characteristic be stored and loaded?

A) Store a versioned strict JSON package resource with an identifier, numeric
points, and canonical content digest; load it once per TESPy session and enforce
snapshot/drift tests. This is the recommended auditable design.

B) Define the characteristic as a private Python tuple constant inside the
TESPy adapter and calculate no content digest.

C) Remove the OpenPinch characteristic and use the installed TESPy default.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Content Validation

This plan contains no Mermaid diagram, ASCII diagram, executable code block,
embedded JSON, or YAML. Markdown headings, lists, option spacing, answer tags,
paths, identifiers, and checkbox syntax were validated before creation.
