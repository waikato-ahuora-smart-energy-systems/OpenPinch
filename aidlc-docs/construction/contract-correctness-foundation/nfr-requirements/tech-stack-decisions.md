# Unit 1 Tech Stack Decisions

- Python and Pydantic remain the implementation and serialization boundary.
- NumPy remains the weighted numeric oracle and unit-aware conversion continues
  through `Value` and `split_report_value`.
- pytest provides example-based regression and contract tests.
- Hypothesis provides PBT-02/PBT-03 with reusable strategies, automatic
  shrinking, and pytest integration.
- CI retains `--hypothesis-seed=20260715`; no new dependency is required because
  Hypothesis is present in `pyproject.toml` and `uv.lock`.

## PBT-09 Compliance

Hypothesis supports constrained composite strategies, shrinking, deterministic
seed replay, and the current pytest runner. Framework selection is compliant.
