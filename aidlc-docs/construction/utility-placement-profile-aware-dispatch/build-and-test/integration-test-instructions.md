# Integration Test Instructions

Run the public workflow and notebook/documentation gates:

```bash
uv run pytest -q tests/application/test_utility_placement.py
uv run pytest -q tests/packaging/test_notebooks.py tests/packaging/test_docs_consistency.py
```

Then execute notebook 19 with the repository Python kernel. Verify the capped
Process case activates at least three feasible named hot levels before
fallback, the Site placement includes a non-isothermal cold span, and the
standard GCC and Total Site Profile cells render without error.
