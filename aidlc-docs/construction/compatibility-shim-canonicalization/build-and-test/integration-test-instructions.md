# Integration Test Instructions

Run the complete suite and both optional executable tutorial profiles:

```bash
uv run pytest -q --hypothesis-seed=20260715
OPENPINCH_TUTORIAL_PROFILES=slow-hpr uv run pytest -q tests/packaging/test_notebooks.py::test_optional_profile_notebooks_execute
OPENPINCH_TUTORIAL_PROFILES=solver uv run pytest -q tests/packaging/test_notebooks.py::test_optional_profile_notebooks_execute
```

Build RTD locally with warnings treated as errors through the repository helper:

```bash
uv run python scripts/build_docs.py --output-dir /private/tmp/openpinch-docs
```
