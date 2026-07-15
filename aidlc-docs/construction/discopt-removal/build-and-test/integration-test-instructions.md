# Experimental Discopt Removal Integration Test Instructions

## Maintained Solver Integration

```bash
uv run pytest -q -m solver
```

Verified result: three passed, one intentionally skipped, and 1,952 deselected.
This exercises maintained solver paths without Discopt.

## Repository and Distribution Contracts

```bash
rg -n -i "discopt" \
  OpenPinch scripts tests docs/developer pyproject.toml uv.lock
zipinfo -1 /tmp/openpinch-discopt-removal-dist/openpinch-0.4.5-py3-none-any.whl
tar -tf /tmp/openpinch-discopt-removal-dist/openpinch-0.4.5.tar.gz
git diff --check
```

Expected result: no active source, test, developer-doc, dependency, or archive
entry refers to Discopt, and the Git diff contains no whitespace errors.

Historical references under `aidlc-docs/` and ignored benchmark results under
`results/` are intentionally preserved as non-executable evidence.
