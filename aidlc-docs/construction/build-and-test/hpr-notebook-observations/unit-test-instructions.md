# Unit and property test instructions

```sh
.venv/bin/python -m pytest -q tests/analysis/heat_pumps tests/application/test_hpr_residual_workflow.py tests/application/test_analysis_contracts.py tests/architecture --hypothesis-seed=20260715
.venv/bin/python -m coverage run --branch --source=OpenPinch -m pytest -q -m 'not solver' --hypothesis-seed=20260715
.venv/bin/python -m coverage report --fail-under=95
```

Retain Hypothesis's shrunk counterexamples and replay seed. Do not relax the
95 percent threshold. The current working tree has three unrelated notebook
preservation failures: test_notebooks_are_valid_nbformat_documents,
test_tutorial_review_preserves_notebook_invariants and
test_notebook_generator_does_not_rewrite_current_notebooks. A separate aggregate
verification excludes exactly these names with pytest -k; the unfiltered result
and reasons remain in build-and-test-summary.md. Image export needs local Chrome
access; it passes outside the restricted sandbox.
