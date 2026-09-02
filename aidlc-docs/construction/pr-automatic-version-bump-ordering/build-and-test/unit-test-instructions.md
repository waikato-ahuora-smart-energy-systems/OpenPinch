# Unit Test Instructions

Run the focused packaging and workflow contracts:

```bash
uv run --no-sync pytest tests/packaging -q
```

Expected result for this change: 123 passed and 3 expected optional-profile
skips. The workflow contract test verifies bump applicability, exact tool
pinning, idempotent version handling, updated-head validation, fork safety, and
aggregate-gate wiring.
