# Build Instructions

From the repository root, use the locked environment:

```bash
uv sync
uv build --out-dir /private/tmp/openpinch-indirect-final-dist
```

The accepted build produced the OpenPinch 0.5.3 source distribution and wheel
from the exact code that passed the complete test suite.
