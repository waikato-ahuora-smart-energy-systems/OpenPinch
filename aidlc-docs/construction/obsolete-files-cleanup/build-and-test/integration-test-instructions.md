# Cleanup integration verification

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/e2e/test_main.py tests/architecture tests/packaging/test_repo_entrypoints.py tests/packaging/test_resources.py --hypothesis-seed=20260715
```

Result: 128 passed in 52.70 seconds. This includes every shipped example input and root workflow smoke tests. Existing Hypothesis seed controls are retained (PBT-08). Full numerical and real-solver suites were not rerun because no application code or input fixtures changed.
