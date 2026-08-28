# Utility Placement Contracts and Pure Model NFR Design Plan

## Design Context

Unit 1 must implement 18 approved NFRs through local patterns and logical
components. It has no independent service process, external system, mutable
state owner, network, persistence, UI, or infrastructure. Infrastructure Design
remains skipped. The next stage after approved NFR Design is Unit 1 Code
Generation planning.

## Plan Steps

- [x] Reconcile the approved Functional Design, 18 Unit 1 NFRs, technology
  decisions, PBT-01 properties, PBT-09 evidence, and package architecture.
- [x] Collect and analyze every NFR Design answer below; add follow-up questions
  for any vague, combined, missing, or contradictory response.
- [x] Obtain explicit approval of the answered NFR Design plan.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-design/nfr-design-patterns.md`.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-design/logical-components.md`.
- [x] Validate coverage of U1-NFR-001 through U1-NFR-018, dependency direction,
  performance evidence, and disabled extension handling.
- [x] Obtain explicit approval before Unit 1 Code Generation planning.

## Question 1 - Resilience Pattern

What fault-handling pattern should the pure model use?

A) Use fail-fast staged validation and complete immutable return values; convert
ordinary candidate invalidity to structured diagnostics; perform no retries,
fallbacks, partial recovery, circuit breaking, or hidden exception suppression

B) Add retry, fallback, or partial-recovery behavior and describe exact trigger
and termination rules after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 2 - Scalability Pattern

How should the design preserve linear growth?

A) Use tuple-backed ordered inputs plus ephemeral indexed lookups, one pass over
period-coordinate bounds, adjacent-only order propagation, and one pass per
encode/decode/verification operation; add no parallelism, streaming protocol,
or retained cache in Unit 1

B) Use another scaling pattern and describe its data structures and complexity
after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 3 - Performance Pattern

How should the implementation meet the 250 ms representative budget?

A) Normalize and convert each public value once, build stable coordinate indices
once per model, reuse tightened effective bounds, avoid repeated Pydantic model
construction inside loops, and enforce the approved p95 benchmark plus linear
scaling check in CI

B) Use a different optimization/benchmark design and describe it after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 4 - Security Pattern

What security implementation pattern applies while the extension is disabled?

A) Use schema and serialization boundaries only: frozen extra-forbid contracts,
finite/dimensional validation, safe primitive JSON, sanitized diagnostics, and
no file/network/code-execution/credential component

B) Add an enabled security pattern or compliance boundary and describe its
components after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 5 - Logical Component Layout

How should production responsibilities be decomposed?

A) Keep one specialist contract module and analysis-owned `errors`,
`normalization`, `bounds`, and `codec` components with an explicit facade;
introduce no queue, cache, worker, circuit breaker, database, or remote client

B) Use different component boundaries and describe exact ownership after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 6 - Observability Pattern

Where should diagnostics and logging occur?

A) Return or raise structured stable context from Unit 1 without emitting logs;
let Unit 2/3 application boundaries decide logging/presentation so pure calls
remain deterministic and library users avoid unsolicited output

B) Emit logs directly from Unit 1 and specify logger names, levels, and events
after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 7 - Unit Conversion Seam

How should conversion remain testable without a second unit system?

A) Centralize conversion in one analysis-owned adapter over existing
`OpenPinch.domain.value.Value.to`, pass canonical quantity metadata explicitly,
and allow focused tests to substitute the adapter callable without mutating the
global Pint registry

B) Call Pint or construct registries independently throughout each component

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 8 - Compatibility Enforcement

How should schema and package compatibility be protected?

A) Add specialist schema/JSON snapshots, root-export assertions, architecture
import rules, wheel/source installed-import smoke, and explicit tests that
existing shared contracts/configuration defaults remain unchanged

B) Rely only on ordinary unit tests and omit dedicated compatibility gates

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)
