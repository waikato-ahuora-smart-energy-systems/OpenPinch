# Unit 2 Business Rules

1. U2-BR-001: Iteration and evaluation limits must be exact positive integers.
2. U2-BR-002: Named public limits override values in `options`.
3. U2-BR-003: Omitted values use `HPRSearchBudget` defaults.
4. U2-BR-004: CoolProp preflight runs before the generic optimiser.
5. U2-BR-005: Preflight covers every normalized VC refrigerant.
6. U2-BR-006: VC+MVR preflight also covers every normalized MVR fluid.
7. U2-BR-007: Preflight diagnostics identify topology, stage, and fluid.
8. U2-BR-008: Search cache keys are exact tuples of finite float coordinates.
9. U2-BR-009: Warm starts are search-evaluated before backend invocation.
10. U2-BR-010: Search evaluation never retains public artifacts.
11. U2-BR-011: A successful finite warm start remains a ranked candidate.
12. U2-BR-012: Backend exhaustion cannot remove a viable warm start.
13. U2-BR-013: Repeated exact points reuse the cached scalar result.
14. U2-BR-014: Returned unsuccessful backend results are candidate-local.
15. U2-BR-015: Raised contract, type, lifecycle, and detachment defects propagate.
16. U2-BR-016: Failure category counts are complete and representatives are capped at 16.
17. U2-BR-017: Representative summaries are single-line detached text.
18. U2-BR-018: No-candidate failures use `HPRTargetingError` and remain `ValueError` compatible.
19. U2-BR-019: Final evaluation occurs outside the search cache in final mode.
20. U2-BR-020: Multiperiod cases share one resolved budget.
21. U2-BR-021: Explicit TESPy selection retains TESPy preflight and never falls back to CoolProp.
22. U2-BR-022: No new runtime dependency or infrastructure component is permitted.
