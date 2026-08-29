# Build Instructions

## Prerequisites

- Python 3.14 and `uv`
- Repository development dependencies already synchronized
- Local solver binaries only for the complete solver-enabled test suite

## Build

Run:

```bash
uv build --out-dir /tmp/openpinch-profile-dispatch-dist
```

Expected artifacts are the OpenPinch source archive and wheel. Both must build
without error, and both archives must contain
`OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`.
