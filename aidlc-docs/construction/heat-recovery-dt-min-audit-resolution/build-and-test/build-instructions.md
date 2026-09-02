# Build Instructions

Run from the repository root with the locked development environment:

```bash
uv sync --frozen --all-groups
uv build
```

Expected artifacts are `dist/openpinch-0.6.3-py3-none-any.whl` and
`dist/openpinch-0.6.3.tar.gz`. The accepted build produced both artifacts.

For an isolated runtime check, create a temporary virtual environment, install
only the wheel and its declared dependencies, change outside the source tree,
and run the `pulp_mill.json` Bleaching threshold example. Assert the specialist
contract import, `at_thermodynamic_limit` status, `dt_min` within `1e-6` of
`58.34505012947355 delta_degC`, JSON round-trip equality, and unchanged cached
ordinary target identity.
