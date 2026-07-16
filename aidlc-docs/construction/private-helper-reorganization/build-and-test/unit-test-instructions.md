# Unit Test Instructions

Run deterministic owner-record and structural checks with:

```text
uv run pytest -q tests/test_classes/test_private_helper_reorganization.py tests/test_package_api_surface.py --hypothesis-seed=20260715
```

Run the complete non-solver matrix with `pytest-cov` and the same seed.

