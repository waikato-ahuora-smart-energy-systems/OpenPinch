# Build Instructions

The change is pure Python and adds no dependency or build-system change.

```bash
uv run python -m compileall OpenPinch/analysis/utility_placement
```

The normal package build remains `uv run python scripts/build_dist.py`.
