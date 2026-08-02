# Integration, Contract, and End-to-End Test Instructions

## Unit Interaction Scenarios

### Workspace identity to presentation export

- Create and load valid named cases through runtime and schema-version-3 bundle
  boundaries.
- Reject invalid names before storage and revalidate at batch export.
- Resolve each valid case destination beneath the selected root.
- Reserve a unique workbook and clean it on writer failure.
- Expected: original result keys, per-case isolation, no path escape, and no
  collision.

### Problem state to analysis

- Read and mutate returned `problem_data` snapshots.
- Run an explicit targeting method and serialize canonical input/results.
- Invoke multiplier mutation on unloaded and lazily prepared problems.
- Expected: internal state is detached, loaded behavior is unchanged, and the
  unloaded path raises the canonical actionable error.

### Exact checkout comparison boundary

- Seed foreign cached OpenHENS modules.
- Enter the requested-checkout scope and validate module origins/capabilities.
- Execute only through the verified injected factory.
- Expected: foreign modules cannot satisfy imports and original interpreter
  state is restored on success and failure.

### Current contract and distribution

- Confirm root exports exactly the two workflow classes.
- Validate serialized HEN mappings through `TargetInput.network`.
- Check current documentation and owner dependencies.
- Build/install the wheel and run its workflow/resource/CLI smoke outside the
  checkout.

## Repository Integration Command

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
```

No service startup, endpoint configuration, database fixture, or cleanup is
required.

## Installed Artifact Smoke

Create a temporary virtual environment with access to the already installed
runtime dependencies, install only the newly built wheel, change outside the
checkout, and run:

```bash
/path/to/venv/bin/python /path/to/OpenPinch/scripts/artifact_install_smoke.py \
  --repo-root /path/to/OpenPinch
```

The smoke must import from site-packages, solve a direct target, construct a
workspace, expose exactly two root names, find packaged tutorials/samples, and
execute CLI help.
