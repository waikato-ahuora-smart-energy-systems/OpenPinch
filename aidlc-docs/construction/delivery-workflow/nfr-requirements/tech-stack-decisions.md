# Delivery technology decisions

- Retain the existing Python 3.14.2 minimum and uv-locked tooling. The delivery
  overhaul does not require a Python migration or new runtime dependency.
- Retain GitHub Actions and package-index trusted publishing. Split concerns
  with reusable validation and small testable Python helpers rather than
  introducing a separate delivery service.
- Retain Hatchling and the existing distribution builder. Preserve exact
  wheel/source archive identity across stages and recovery attempts.
- Retain pytest, Coverage.py, Ruff, and Hypothesis. `pyproject.toml` already
  declares Hypothesis in development dependencies; it supports the required
  structured strategies, shrinking, reproducible seeds, and pytest integration.
- Use injectable clocks and transport boundaries for deterministic delivery
  tests. Tests simulate responses and state; they never exercise actual
  package publication as part of regression verification.
- Add a pinned workflow-linting tool in development/CI tooling as needed.
  Select its concrete version during implementation from authoritative
  upstream documentation; do not introduce it into package runtime dependencies.
- Keep credentials and publishing authorization in existing external controls.
  Do not add personal access tokens merely to make workflow events retrigger.

## PBT compliance

PBT-09 is compliant: the existing Python/Hypothesis stack is selected,
declared, and suitable for all properties identified in Functional Design.
PBT-01 through PBT-08 and PBT-10 are N/A to this NFR Requirements stage;
their design and implementation obligations remain in the functional design.
Security and Resiliency extensions remain disabled. This does not waive the
explicit delivery security and reliability requirements.
