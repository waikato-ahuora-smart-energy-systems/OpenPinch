# Build Instructions

## Prerequisites

- Build tool: uv with Hatchling through the PEP 517 build interface.
- Runtime: CPython 3.14.2 or later.
- Core dependencies: NumPy, Pint, pandas, CoolProp 8 or later, Pydantic, and
  SciPy.
- HPR optional dependency: install the `tespy` extra for TESPy target and map
  generation. The verified environment uses TESPy 0.11.2.
- Development dependencies: pytest, Hypothesis, coverage, Ruff, Sphinx,
  `build`, and Hatchling from the locked development group.
- External solver profile: locally configured IPOPT, CBC, Bonmin, Couenne,
  APOPT, MojoPSE, and AMPL function libraries for solver-marked repository
  tests. They are not required by HPR targeting or map generation.
- Disk: temporary space for a source archive, wheel, and two isolated virtual
  environments.

## Build Steps

### 1. Synchronize Dependencies

```bash
uv sync --all-extras --group dev
```

### 2. Verify the Runtime and Optional Boundaries

```bash
uv run python --version
uv run python scripts/optional_install_smoke.py tespy
```

### 3. Build Source and Wheel Artifacts

Use a clean output directory:

```bash
uv build --out-dir dist
```

### 4. Verify Build Success

- Expected artifacts for version 0.6.4 are
  `dist/openpinch-0.6.4.tar.gz` and
  `dist/openpinch-0.6.4-py3-none-any.whl`.
- Both archives must contain the versioned HPR schema, heat-pump and
  refrigeration fixtures, compressor characteristic, all performance-map
  modules, target record/basis modules, and public accessor integration.
- Wheel metadata must expose a `tespy` extra requiring TESPy 0.10.1.post2 or
  later.
- Archive members must be unique and resource bytes must match the checkout.

## Installed Artifact Verification

Install the wheel into two clean environments outside the checkout. Run:

```bash
python scripts/artifact_install_smoke.py --surface core
python scripts/artifact_install_smoke.py --surface tespy
```

The core environment must not contain TESPy. The TESPy environment must execute
the explicit public target, winning-record, target-derived basis, and minimal
performance-map workflow. Both imports must resolve from site-packages.

## Troubleshooting

### Dependency or Build Frontend Failure

Synchronize the lock and rerun the build from the repository root. In a
network-restricted environment, ensure uv can access its existing package
cache. Do not weaken the optional dependency boundary to make the build pass.

### Missing Resource or Signature Mismatch

Run the packaging, resource, and cold-import tests. Compare the source, sdist,
and wheel resource digests, then rebuild from a clean output directory. Do not
manually patch a built archive.
