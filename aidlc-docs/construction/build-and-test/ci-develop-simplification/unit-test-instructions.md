# Decision and contract tests

Run `.venv/bin/pytest -q tests/packaging/test_reuse_develop_ci.py tests/packaging/test_packaging_metadata.py`.
Expected result: 66 passed. The generated required-job mutation property uses
seed 20260918; failures must prevent reuse rather than suppress validation.
