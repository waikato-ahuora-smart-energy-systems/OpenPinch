# Unit Test Instructions

Run the feature and ordinary-cascade regression selections:

```bash
uv run pytest tests/analysis/test_heat_recovery_dt_min.py \
  tests/analysis/test_heat_recovery_dt_min_properties.py \
  tests/application/test_heat_recovery_dt_min.py \
  tests/application/test_heat_recovery_dt_min_api.py \
  tests/application/test_heat_recovery_dt_min_properties.py \
  tests/contracts/test_heat_recovery_dt_min.py -q

uv run pytest tests/analysis/test_direct_targeting.py \
  tests/analysis/test_problem_table.py \
  tests/domain/test_problem_table_kernels.py -q
```

The accepted feature selection passed 104 tests. The accepted ordinary
cascade selection, including the inverse analysis tests, passed 122 tests.

Run lint and changed-file formatting checks:

```bash
uv run ruff check .
uv run ruff format --check <changed-python-files>
git diff --check
```

Repository-wide formatting currently identifies 14 pre-existing files outside
this correction; every changed Python file passes the format check.
