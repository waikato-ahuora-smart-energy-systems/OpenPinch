# Unit Test Instructions

Run the utility-placement analytical and property suite:

```bash
uv run pytest -q tests/analysis/utility_placement
```

The suite covers temperature-only codec round trips, ordering and bounds,
candidate-local allocation, fallback penalties, balanced-composite entropy,
evaluation accounting, and bounded performance. Any failure blocks completion.
