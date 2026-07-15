# Experimental Discopt Removal Build Instructions

## Prerequisites

- Python 3.14.2 or newer.
- The locked uv environment for normal tests, Sphinx, and packaging.
- Couenne and IPOPT available on `PATH` for solver-marked tests.
- No Discopt package or Rust toolchain is required.

## Build Steps

### Documentation

```bash
uv run sphinx-build -E -a -W -b html docs /tmp/openpinch-docs-discopt-removal
```

Expected result: all 58 documentation sources build without warnings and the
fresh output contains no Discopt references.

### Distributions

```bash
uv run python scripts/build_dist.py --output-dir /tmp/openpinch-discopt-removal-dist
```

Expected artifacts:

- `openpinch-0.4.5-py3-none-any.whl`
- `openpinch-0.4.5.tar.gz`

Inspect both archive listings and confirm they contain no `_discopt.py`,
`benchmark_hen_solvers.py`, or `hen_benchmarks.py` entries.

## Troubleshooting

- If uv cannot initialize its user cache inside a sandbox, rerun with access to
  the existing uv cache rather than changing project dependencies.
- If solver tests cannot find Couenne or IPOPT, expose the configured IDAES
  solver directory on `PATH` and rerun only the solver-marked suite.
