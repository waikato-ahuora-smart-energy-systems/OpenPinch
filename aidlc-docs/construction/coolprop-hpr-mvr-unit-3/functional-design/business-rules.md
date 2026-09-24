# Unit 3 Business Rules

1. U3-BR-001: Stage lift must be finite, positive, and at most 200 degC.
2. U3-BR-002: Stage pressure ratio must be finite, greater than one, and at most 20.
3. U3-BR-003: Both compression target forms may not be supplied together.
4. U3-BR-004: Bound rejection occurs before stage iteration.
5. U3-BR-005: Required property failures raise `DirectGasMVRStageError`.
6. U3-BR-006: Public error text is bounded and omits raw CoolProp call arguments.
7. U3-BR-007: The original property failure is available only as the chained cause.
8. U3-BR-008: A failed requested stage prevents return of all partial replacements.
9. U3-BR-009: Injection saturation failure may use only `dry_stage`.
10. U3-BR-010: Optional profile saturation failure may use only `reduced_profile`.
11. U3-BR-011: Every applied fallback is represented by an immutable diagnostic.
12. U3-BR-012: Unexpected non-property exceptions propagate unchanged.
13. U3-BR-013: Normal successful stage accounting and units remain unchanged.
14. U3-BR-014: Notebook examples may catch typed physical infeasibility, not arbitrary exceptions.
