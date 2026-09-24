# Unit 1 NFR Requirements

## Scope

These requirements constrain Unit 1 candidate normalization, evaluation modes,
detached contracts, result finalization, and transaction copy safety.

## Performance and Scalability

| ID | Requirement | Verification |
|---|---|---|
| NFR-U1-001 | Penalty normalization is O(n) time and O(n) output memory for n terms, with one numeric conversion and one stable flattening pass. | Structural call-count test and 100,000-term benchmark on the supported test profile. |
| NFR-U1-002 | Recursive detachment validation is O(v + e) over the public object graph using identity tracking. | Generated nested graphs and visit-count assertions. |
| NFR-U1-003 | Search mode constructs no public streams, simulation record, debug figure, or persistent engine model. | Artifact factory spies remain at zero during search evaluations. |
| NFR-U1-004 | Final artifact construction occurs only for final-mode points and at most once per final evaluation. | Deterministic constructor call-count tests. |
| NFR-U1-005 | Unit 1 helpers retain no cross-call cache or mutable global state. | Isolation properties across sequential calls. |
| NFR-U1-006 | On the supported CI profile, normalizing 100,000 finite terms and validating the packaged detached result each complete within 250 ms independently. | Marked deterministic performance test with documented environment and non-flaky guard. |

## Reliability and Atomicity

| ID | Requirement | Verification |
|---|---|---|
| NFR-U1-007 | Pure transformations are deterministic for identical inputs and configuration. | Repeated-call properties and fixed-seed examples. |
| NFR-U1-008 | Candidate-local physical failure and fatal contract/lifecycle failure are never conflated. | Generated outcome sequences and explicit fatal examples. |
| NFR-U1-009 | Public outputs contain no live engine object at any depth and are deep-copy safe. | Recursive forbidden-type scan plus real CoolProp deep-copy regression. |
| NFR-U1-010 | A failed validation or deep copy causes no application-state mutation. | Transaction rollback/unchanged-state test. |
| NFR-U1-011 | Nested multiperiod detachment is complete and preserves period identity/order. | Inductive property over generated period collections. |
| NFR-U1-012 | No automatic retry hides a deterministic Unit 1 failure. | Exception and call-count assertions. |

## Boundedness and Defensive Handling

| ID | Requirement | Verification |
|---|---|---|
| NFR-U1-013 | Diagnostic summaries and representative collections are length-bounded; raw tracebacks and exception objects are excluded. | Boundary properties and forbidden-field inspection. |
| NFR-U1-014 | Unknown arbitrary public-object types fail closed with a concise typed error. | Generated forbidden-object injection. |
| NFR-U1-015 | Budget values reject booleans, coercive strings/floats, zero, negatives, NaN, and infinity. | Domain strategy and explicit boundaries. |
| NFR-U1-016 | Simulation assumptions and record strings are non-empty and bounded; loop/stage collections are finite and topology-consistent. | Contract validation properties. |

## Compatibility and Usability

| ID | Requirement | Verification |
|---|---|---|
| NFR-U1-017 | Existing public field names and single-stage record semantics remain compatible; `model` remains readable as `None`. | Existing regression suite and compatibility examples. |
| NFR-U1-018 | `HPRTargetingError` remains catchable as `ValueError` and exposes a typed immutable summary. | Inheritance and payload tests. |
| NFR-U1-019 | Public error messages are concise, stable at the reason-code level, and do not require parsing CoolProp text. | Exact reason-code and bounded-message assertions. |
| NFR-U1-020 | No new public root export is added unless required by an existing contract-export convention. | Public API inventory test. |

## Maintainability and Quality

| ID | Requirement | Verification |
|---|---|---|
| NFR-U1-021 | One shared penalty normalizer and one finalization policy serve all HPR topologies. | Architecture/import and usage tests. |
| NFR-U1-022 | Changed production code achieves at least 95 percent statement and branch coverage in the focused Unit 1 profile. | Coverage gate. |
| NFR-U1-023 | Every critical path has explicit regression examples and applicable generated properties. | Test inventory mapped to U1-P1 through U1-P12. |
| NFR-U1-024 | Ruff, repository formatting, Python compilation, patch hygiene, and warning-strict documentation pass. | Standard repository gates. |
| NFR-U1-025 | No runtime dependency, infrastructure component, network service, persistence system, or deployment artifact is introduced. | Dependency and package inspection. |

## Availability and Operations

Traditional uptime, horizontal scaling, disaster recovery, failover, monitoring,
and alerting requirements are N/A because Unit 1 is synchronous library code
inside the caller's process. Operational reliability is expressed through
deterministic failure, atomic transaction behavior, bounded memory, and
actionable typed errors.

## PBT Compliance

- **PBT-09**: compliant. Hypothesis with pytest is the selected existing
  framework; it supports constrained strategies, shrinking, seeds, and CI.
- PBT-02 through PBT-05, PBT-07, PBT-08, and PBT-10 are mandatory in Code
  Generation for the Unit 1 properties.
- PBT-06 is N/A to Unit 1 pure/immutable components; the transaction rollback
  is covered by example and generated input tests rather than a mutable command
  state machine.
- No blocking PBT finding exists.
