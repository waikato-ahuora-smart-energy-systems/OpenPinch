# Performance Test Instructions

## Applicability

No performance contract or hot-loop algorithm changed. The correction adds
constant-size problem-state snapshots, linear work over a utility's segment
count, and feasibility checks over returned optimizer candidates and supplied
constraints.

## Regression signal

Use the complete suite duration as the bounded regression signal:

```bash
uv run pytest
```

The verified configured-solver run completed in 356.52 seconds. Investigate a
material repeatable increase under the same machine, solver, and cache state;
single-run timing variation is not a failure by itself.
