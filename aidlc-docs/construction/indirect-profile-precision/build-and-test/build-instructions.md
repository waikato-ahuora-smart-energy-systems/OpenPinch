# Build Instructions

From the repository root, use the locked environment:

```bash
uv sync
uv build --out-dir /private/tmp/openpinch-precision-dist
```

The accepted build produced the OpenPinch 0.5.3 source distribution and wheel
from the exact precision-preserving implementation tested in this workflow.
