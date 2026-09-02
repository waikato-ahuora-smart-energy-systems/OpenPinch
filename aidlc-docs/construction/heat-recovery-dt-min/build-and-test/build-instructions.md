# Build Instructions

From the repository root:

```bash
uv sync --group dev
uv run python scripts/build_dist.py --output-dir <artifact-directory>
```

The build produces one wheel and one source distribution. Use a clean
temporary artifact directory for release verification so existing `dist/`
contents are not involved.
