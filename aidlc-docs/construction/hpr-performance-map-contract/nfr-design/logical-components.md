# Unit 1 Logical Components

## Component inventory

| Component | Responsibility | Runtime dependency |
|---|---|---|
| LC-01 Contract values | Strict immutable request, units, point, and map shapes | Pydantic and standard library |
| LC-02 Primitive/JSON guards | Finite values, identities, recursive JSON domain, detachment | Standard library |
| LC-03 Semantic map validator | Mode, units, energy, capacity, COP, curve, and order invariants | LC-01 and LC-02 |
| LC-04 Canonical JSON formatter | Stable UTF-8 resource text and byte generation | Standard-library JSON |
| LC-05 Schema/fixture factory | Build schema and two validated representative maps | LC-01 through LC-04 |
| LC-06 Contract resource catalog | Read schema/fixtures from installed package without path assumptions | `importlib.resources` |
| LC-07 Domain test strategies | Generate valid requests, provenance, points, curves, and maps | Hypothesis; tests only |
| LC-08 Independent schema verifier | Validate schema and fixtures without OpenPinch model decoding | `jsonschema`; tests only |
| LC-09 Architecture/package verifier | Enforce imports, resources, distributions, and installed smoke | pytest/build tooling; tests only |

## LC-01: Contract values

LC-01 is the only authoritative definition of external field names and types.
It contains:

- `JsonValue`;
- `HprPerformanceMapRequest`;
- `HprPerformanceMapUnits`;
- `HprPerformancePoint`; and
- `HprPerformanceMap`.

All model configuration is strict, frozen, and non-finite rejecting. The map
coordinates its child point and curve checks; no second service revalidates a
weaker copy of the contract.

**NFR ownership**: NFR-U1-001 through NFR-U1-003, NFR-U1-007, NFR-U1-015.

## LC-02: Primitive and recursive JSON guards

LC-02 provides small pure helpers for:

- exact nonempty-string handling without lossy coercion;
- finite positive/nonnegative/fraction checks;
- canonical request tuple sorting and duplicate rejection;
- recursive string-keyed JSON provenance validation; and
- detached JSON-compatible copies.

These helpers do not know about engines, HPR targets, resources, or external
consumer types.

**NFR ownership**: NFR-U1-004, NFR-U1-013, NFR-U1-014.

## LC-03: Semantic map validator

LC-03 performs one indexed point scan and grouped curve scan. It applies:

- closed units, modes, capacity basis, COP convention, and topology;
- unique point and coordinate identities;
- one temperature pair and ascending unique loads per curve;
- global deterministic point order;
- energy balance, useful-duty/reference-capacity consistency, and COP; and
- separate nonnegative tolerances.

It reports invariant-focused locations/messages and returns only a fully valid
immutable map.

**NFR ownership**: NFR-U1-002, NFR-U1-004, NFR-U1-007 through NFR-U1-009.

## LC-04: Canonical JSON formatter

LC-04 accepts only plain JSON-compatible content already produced by a valid
contract or schema factory. It applies the one canonical formatting policy and
returns detached text/bytes. It neither opens files nor imports package
resources.

**NFR ownership**: NFR-U1-008, NFR-U1-014, NFR-U1-015.

## LC-05: Schema and fixture factory

LC-05 builds:

- the authoritative map JSON Schema;
- one three-or-more-point heat-pump map; and
- one three-or-more-point refrigeration map.

Each fixture is first constructed as a valid LC-01 map, then serialized by
LC-04. The factory exposes explicit check and regeneration behavior to
development tooling. It performs no work during ordinary imports.

**NFR ownership**: NFR-U1-005, NFR-U1-006, NFR-U1-008, NFR-U1-015,
NFR-U1-018, NFR-U1-019.

## LC-06: Contract resource catalog

LC-06 defines the three allowed resource names and reads through
`importlib.resources`. Unknown names fail closed. Parsed mappings are newly
allocated and detached. Resource lookup works in a checkout and installed
archive without exposing an assumed filesystem path.

**NFR ownership**: NFR-U1-005, NFR-U1-006, NFR-U1-018, NFR-U1-019.

## LC-07: Domain test strategies

LC-07 is test-only and generates values that satisfy mode-specific physics and
curve topology by construction. Strategies cover boundary values, nested JSON,
multiple curves, request permutations, and tolerances while remaining small
enough to shrink effectively.

It centralizes shared strategies and never calls a thermodynamic engine.

**NFR ownership**: NFR-U1-016, NFR-U1-017; PBT-02, PBT-03, PBT-07,
PBT-08, PBT-10.

## LC-08: Independent schema verifier

LC-08 is test-only. It validates the committed schema's meta-schema and checks
plain parsed golden JSON against that schema using development-only
`jsonschema`. It does not import the OpenUtility package and is not shipped as a
runtime dependency.

**NFR ownership**: NFR-U1-002, NFR-U1-011, NFR-U1-015, NFR-U1-016; PBT-05.

## LC-09: Architecture and package verifier

LC-09 owns:

- subprocess imports with TESPy and external optimizer packages blocked;
- static proof that Unit 1 does not import CoolProp or another engine directly;
- static forbidden-import checks for Unit 1 production modules;
- existing root-export and HPR regression checks;
- resource drift and size checks;
- source/wheel inventory checks; and
- isolated installed-wheel resource loading and validation.

**NFR ownership**: NFR-U1-003, NFR-U1-005, NFR-U1-006, NFR-U1-010 through
NFR-U1-012, NFR-U1-019, NFR-U1-020.

## Dependency direction

| Consumer | May depend on | Must not depend on |
|---|---|---|
| LC-01 | Pydantic, LC-02 pure helpers | LC-04 through LC-09, runtime HPR, engines, consumers |
| LC-02 | Standard library | All other Unit 1 components |
| LC-03 | LC-01 and LC-02 | Resources, tests, engines, application layer |
| LC-04 | Standard-library JSON | Engines, application layer, consumer packages |
| LC-05 | LC-01 through LC-04 | LC-06 runtime reader, engines, application layer |
| LC-06 | Standard resources/JSON | LC-05 regeneration, engines, application layer |
| LC-07 | LC-01, Hypothesis | Production imports from tests |
| LC-08 | LC-06, `jsonschema` | Runtime package dependency on `jsonschema` |
| LC-09 | Production/resource public boundaries and test/build tools | Production dependency on any verifier |

The production dependency graph is acyclic. Test and development components
depend inward on production components; no production component imports tests,
regeneration tooling, engines, or OpenUtility.

## Data flow

### Runtime map flow

Raw JSON-like data enters LC-01/LC-02 structural checks, continues through
LC-03 semantic validation, and becomes one immutable map. LC-04 may serialize a
detached representation. No file write or engine call occurs.

### Development resource flow

LC-05 constructs validated fixture maps and generated schema, LC-04 formats
them, and explicit development tooling checks or regenerates the three package
resources. LC-08 independently validates the committed outputs.

### Installed consumer flow

LC-06 reads committed resources through the installed package. A caller may
parse them as plain JSON or validate fixture data through LC-01. OpenUtility may
vendor/read the JSON separately but has no Python edge to LC-06.

## Scaling and failure behavior

- Contract operations are synchronous and local.
- Point/curve/provenance work is linear in input size.
- There is no shared cache or mutable singleton, so calls do not contend or
  require locks.
- Validation failures are deterministic and are not retried.
- Resource absence or corruption is a packaging/test failure, not a silent
  request fallback.
- Schema/fixture regeneration is an explicit development mutation and never a
  runtime recovery action.

## Logical-component verification matrix

| Component | Focused evidence | Broader evidence |
|---|---|---|
| LC-01 | Construction, frozen values, JSON round trip | Existing contract regressions |
| LC-02 | Boundary/provenance/detachment properties | Hostile input examples |
| LC-03 | Every physical/topology business rule | 10,000-point performance case |
| LC-04 | Canonical byte snapshots and repeated process output | Distribution byte equality |
| LC-05 | Schema and two fixture generation | Drift check |
| LC-06 | Source and installed resource reads | Wheel/sdist smoke |
| LC-07 | Strategy health, shrinking, boundary generation | Fixed-seed CI PBT |
| LC-08 | Meta-schema and plain fixture validation | External consumer fixture evidence |
| LC-09 | Blocked imports and forbidden edges | Full quality/release gate |

## Infrastructure disposition

Infrastructure Design remains N/A for Unit 1. LC-01 through LC-06 are in-process
library/resource components, and LC-07 through LC-09 are development/test
components. No cloud, host, container, queue, cache, database, secret, monitor,
or deployment topology is introduced.
