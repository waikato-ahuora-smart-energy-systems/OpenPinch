# Performance Test Instructions

Run:

```bash
uv run pytest -q tests/analysis/utility_placement/test_performance.py
```

The pure-model 100-period pipeline must remain inside its existing 250 ms p95
gate. Exact application replay must honor the configured optimizer evaluation
limit and terminate deterministically; no universal wall-clock limit is imposed
because ordinary hierarchy targeting cost varies by study.
