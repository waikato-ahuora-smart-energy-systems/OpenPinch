# Performance Test Instructions

## Post-construction audit clarification

Use `tests/application/test_coolprop_hpr_audit.py` for real public calls with
3 iterations and 100 search evaluations (150 for multi-stage records). Each
search is instrumented without replacing the optimizer; the ten scalar workflow
cases assert at most 100 unique evaluations and a generous 120-second smoke
ceiling per case. This is not a production latency SLA.

The hard evaluation cap covers warm starts, restarts and serial polishing within
one search. Optional Carnot screening has its own search; final artifact
reevaluations are additional. Shared-period objectives may perform several
period solves in one evaluation. Cache hits do not count as fresh evaluations.

No server throughput or concurrent-user benchmark applies to this local Python
library. Performance acceptance is algorithmic and bounded:

- Public iteration/evaluation budgets must map exactly to optimizer limits.
- Warm starts execute before backend candidates and are retained if viable.
- Repeated exact candidate coordinates execute the thermodynamic objective once.
- Search mode must not retain figures, engine objects, or simulation records.
- Preflight must reject unsupported fluids before optimizer entry.

Run the focused budget/caching properties with:

```bash
uv run pytest tests/analysis/heat_pumps/test_hpr_search_reliability.py -q
```

Run the full repository suite to detect runtime or memory regressions in the
existing performance gates. No external load generator or service environment
is required.
