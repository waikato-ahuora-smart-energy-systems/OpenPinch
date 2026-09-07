# Obsolete files cleanup verification

- Deleted tracked June coverage JSON and the unreferenced retired-API debug notebook.
- Removed disposable Python, pytest, Ruff and Sphinx caches, Finder metadata, root coverage database and timing log, and both local 0.6.3 distribution archives.
- Added `/coverage.json` to `.gitignore`; ignore verification passed.
- 128 focused tests passed; wheel and sdist 0.6.5 built in `/tmp`.
- Wheel retains all 19 maintained tutorials; entrypoint exists; removed artifacts are absent.
- Ruff and `git diff --check` passed. Every selected removal path is absent.
- Notebook 19 matches its pre-cleanup SHA-256. Notebooks 01 and 02 changed independently during the session and were left untouched.
- Preserved old workbook versions, fixtures and result baselines, research/private data, environments, OpenHENS checkout, Hypothesis examples, and workflow history.
- No source/API changes, commits, pushes, or deployment.

## Extension compliance

Security and Resiliency: disabled and skipped. PBT-01, PBT-02, PBT-03, PBT-04, PBT-05, PBT-06, PBT-07 and PBT-10: N/A because no business logic or tests changed. PBT-08: compliant, verification uses seed 20260715. PBT-09: compliant, existing Hypothesis dependency retained. No blocking findings.

## Second pass

Removed 54 unused `examples/results/r_*.json` workbook snapshots and
`examples/review/other/r_new_example_3.json`: 55 tracked files, 744,812 bytes.
Current e2e tests solve the input fixtures directly; the migration baseline
confirms these snapshots are not its contract oracle. Specialist fixtures and
all source workbooks and input fixtures remain unchanged.

The existing regeneration helper now creates its missing results directory. A
controlled temporary-directory smoke verified first and repeated invocation.
Both generated-output locations are ignored. Ruff and patch hygiene passed.

Verification command:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/e2e/test_main.py tests/adapters/test_workbook.py --hypothesis-seed=20260715
```

Result: 79 passed in 13.10 seconds. The package build from the first pass still
covers the unchanged distribution boundary; second-pass changes are outside
that boundary. Notebook 05 changed independently during this pass; those edits
were preserved, as were all other notebook edits. Both historical Excel releases
remain because the optional removal question received no answer.

Extension settings unchanged: Security and Resiliency disabled; existing PBT
seed retained. The helper fix is covered by a filesystem smoke, with no new
algorithm, serialization, or stateful application policy requiring PBT.
