# Unit Test Instructions

Run the specialist analytical and property suite:

```bash
uv run pytest -q tests/analysis/utility_placement
```

The suite covers coordinate contracts, duty conservation, encode/decode round
trips, cap-aware allocation, entropy scaling, candidate ranking, and evaluation
accounting. Any failure blocks release.
