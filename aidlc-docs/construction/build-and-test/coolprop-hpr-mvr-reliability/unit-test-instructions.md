# Unit Test Instructions

Run the complete repository suite with the configured local solver executables:

```bash
uv run pytest
```

For rapid HPR/MVR iteration, run:

```bash
uv run pytest tests/analysis/heat_pumps tests/contracts -q
```

The required result is zero failures. Enabled deterministic Hypothesis tests
cover penalty normalization, budgets, warm-start caching, topology records,
direct-MVR bounds, errors, and fallback invariants. Review normal pytest output;
this workflow does not create a persistent report file.
