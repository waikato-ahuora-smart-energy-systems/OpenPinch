# Integration verification

Run `.venv/bin/pytest -q tests/packaging` to cover packaging and workflow contracts.
Observed result: 167 passed, 3 skipped. Decision tests mock API responses;
gate tests execute the actual workflow bash with fresh and reused result sets.

After an authorized push, check that CI Develop runs all validation jobs. Once
it succeeds, a develop-to-main PR with an identical merge tree should show a
successful reuse decision and test status, skip shared jobs, and still run
solver and release checks. A PR with unavailable proof must run shared checks.
This remote verification has not been performed locally.
