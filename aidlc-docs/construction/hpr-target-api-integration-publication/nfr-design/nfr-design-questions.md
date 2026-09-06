# Unit 3 NFR Design Questions

Please answer all five questions. Each addresses one mandatory NFR Design
category and confirms the implementation pattern to carry into construction.

## Question 1
Which resilience pattern should govern TESPy targeting candidate failures?

A) Do not retry or fall back. Treat a recoverable candidate-local failure as
one infeasible candidate, restore a clean evaluator state, and continue; abort
the request for dependency, topology, lifecycle, restoration, or cleanup
failures.

B) Retry every failed TESPy candidate once in the same evaluator before marking
it infeasible, while retaining fatal-request aborts.

C) Fall back to CoolProp for a failed TESPy candidate so the optimizer can
continue with a complete candidate set.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 2
Which scalability and concurrency pattern should Unit 3 expose?

A) Keep each target/map call sequential and isolated, add no internal scheduler
or global lock, and document separate processes as the supported way to run
independent calls concurrently.

B) Add an internal process pool that automatically evaluates TESPy optimizer
candidates in parallel.

C) Add a shared thread pool and one process-global TESPy lock.

X) Other (please describe after [Answer]: tag below)

[Answer]: A. However, the optimiser may run simulations in parallel.

## Question 3
Which performance pattern should implement the approved candidate-cache and
release thresholds?

A) Use the approved 512-entry exact-key call-local LRU, structural solve-count
checks, a 64-MiB traced Python cache limit, and the marked 300-second public
TESPy target-and-map smoke with elapsed trend reporting.

B) Replace the fixed LRU with an adaptive unbounded cache and rely on elapsed
time only.

C) Remove caching and memory checks, retaining only the 300-second real-engine
smoke.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 4
Which security and integrity pattern is appropriate for this in-process library
feature?

A) Keep the Security extension disabled and add no authentication or sandbox
subsystem; enforce strict inputs, optional-dependency isolation, bounded
sanitized diagnostics, no temporary-path/object disclosure, and artifact
integrity checks as ordinary library safeguards.

B) Enable the full Security Baseline extension and redesign Unit 3 around its
applicable requirements before construction.

C) Add an in-process allowlist that permits only named refrigerants, even when
the installed property backend supports other pure fluids or mixtures.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 5
Which logical-component boundary should contain Unit 3?

A) Keep it as in-process components within existing application, contracts,
domain, analysis, documentation, and test owners: selector/replay adapter,
evaluator factory/session, bounded cache, normalized validator, target-record
builder, map-basis builder, explicit map bridge, and release gates.

B) Create a separately installed OpenPinch TESPy targeting plugin that owns the
selector, target record, and map bridge.

C) Create a persistent local simulation service and disk cache used by both
OpenPinch and OpenUtility.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Content Validation

This question file contains no Mermaid diagram, ASCII diagram, executable code
block, JSON, or YAML. Markdown headings, choices, blank-line separation,
special characters, units, and answer tags were checked before creation.
