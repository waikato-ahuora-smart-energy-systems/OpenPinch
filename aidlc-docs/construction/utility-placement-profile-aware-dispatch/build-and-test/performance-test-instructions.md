# Performance Test Instructions

Run:

```bash
uv run pytest -q tests/analysis/utility_placement/test_performance.py
```

The representative 100-period pure-model pipeline must remain within its
250 ms p95 gate. Deterministic warm-start count is adaptively bounded as the
period-specific coordinate vector grows; no domain count ceiling is imposed.
