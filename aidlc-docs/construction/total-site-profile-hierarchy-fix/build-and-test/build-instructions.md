# Build Instructions

From the repository root, use the locked environment:

```bash
uv sync
uv build --out-dir /tmp/openpinch-total-site-build
```

The accepted build produced `openpinch-0.5.3.tar.gz` and
`openpinch-0.5.3-py3-none-any.whl` in an isolated temporary directory.

