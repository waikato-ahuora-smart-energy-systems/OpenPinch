# Unit 1 NFR Design Patterns

- **Policy table**: one centralized mapping separates weighted, peak, derived,
  consensus, and optional behavior.
- **Validate then transform**: alignment and weights are validated before output
  construction.
- **Copy-on-output**: source Pydantic models are dumped and revalidated into new
  records, preserving input immutability.
- **Deterministic order**: first-output target order and first-seen utility order
  are retained.
- **Actionable failure**: exceptions name the public field or target mismatch.
- **Property plus example tests**: generalized invariants supplement concrete
  regression scenarios.

Resilience, horizontal scaling, queues, caches, circuit breakers, and cloud
security patterns are N/A for this pure in-process transformation.
