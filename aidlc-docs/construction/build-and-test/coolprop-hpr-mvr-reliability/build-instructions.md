# Build Instructions

## Prerequisites

- Python 3.14.2 or newer.
- The locked `uv` development environment with Hatchling and Sphinx.
- CoolProp 8 or newer; TESPy is needed only for the optional-backend gate.
- IPOPT, CBC, Bonmin, Couenne, APOPT, and MojoPSE executables for the complete
  solver-marked repository suite.

## Build Steps

From the repository root:

```bash
UV_CACHE_DIR=/private/tmp/openpinch-uv-cache uv sync --frozen
UV_CACHE_DIR=/private/tmp/openpinch-uv-cache uv run python -m sphinx \
  -W --keep-going -b html docs /private/tmp/openpinch-hpr-mvr-docs
UV_CACHE_DIR=/private/tmp/openpinch-uv-cache uv build \
  --out-dir /private/tmp/openpinch-hpr-mvr-dist
```

Successful output contains one wheel and one source distribution. Inspect both
for the HPR contracts, CoolProp adapters/preflight, direct-MVR implementation,
sample cases, and notebooks 09 through 11. No live engine state or build cache
belongs in either artifact.

## Troubleshooting

- If the cache is sandbox-restricted, keep `UV_CACHE_DIR` on `/private/tmp`.
- If strict documentation fails, correct every warning rather than suppressing it.
- If an optional solver is unavailable, report that gate separately; do not count
  a skipped mandatory check as a pass.
