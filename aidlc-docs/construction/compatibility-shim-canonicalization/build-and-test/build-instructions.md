# Build Instructions

From the repository root:

```bash
uv build --out-dir /private/tmp/openpinch-dist
```

The expected artifacts are the OpenPinch 0.5.2 wheel and source distribution.
Install the wheel into a fresh environment before running
`scripts/artifact_install_smoke.py` outside the checkout.
