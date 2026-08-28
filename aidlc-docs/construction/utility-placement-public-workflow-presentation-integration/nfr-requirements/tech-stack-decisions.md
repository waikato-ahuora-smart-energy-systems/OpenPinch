# Unit 3 Technology Decisions

- Extend the existing Python target and workspace accessor classes in place.
- Use Pydantic Unit 1 contracts and the Unit 2 specialist service unchanged.
- Reuse `PinchProblem.to_problem_json()` to create isolated execution copies.
- Reuse pandas for summary frames and existing report conventions for text output.
- Reuse `CaseBatchResult` and mapping proxies for ordered batch isolation.
- Reuse the canonical Python notebook generator, resource manifest, pytest, Hypothesis, coverage, Ruff, Hatchling, and build tooling.
- Add no dependency and no CLI integration.
