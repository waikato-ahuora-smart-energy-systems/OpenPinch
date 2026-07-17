# Package Architecture Modernization Baseline

## Source Baseline

- Git revision: `f93f34abc34c8e8d17f95ebe8adc15172938f347`.
- Package size: 60,201 physical Python lines across 251 files.
- Protected external import:
  `from OpenPinch.main import pinch_analysis_service`.
- Protected signature:
  `pinch_analysis_service(data: Any, project_name: str = "Project") -> TargetOutput`.
- `OpenPinch.main` cold import: 132 loaded `OpenPinch` modules in 1.375103
  seconds on the development machine. This is a diagnostic baseline, not a
  timing threshold.

## Behavioural Baselines

- Main external contract: 59 caller-level tests after activating the contract
  assertions in `tests/e2e/test_main.py`.
- Segmented streams: 46 tests.
- Multiperiod summaries: 9 tests.
- Multiperiod HPR: 13 tests.
- HEN contracts and segmented streams: 83 tests.
- Existing canonical HPR, HEN, notebook, documentation, packaging, and complete
  non-solver suites remain the final integration baselines defined in the
  approved checklist and existing Build and Test artifacts.

## Contract Assertions

The main contract suite protects:

- parameter names, kinds, annotations, and default project name;
- `TargetOutput` model identity by model name rather than internal module path;
- validation error type and location for missing stream input;
- top-level and target-result serialized field ordering;
- representative target values and graph keys;
- validated Pydantic input handling;
- successful finite targets for every shipped example; and
- importability when optional feature packages are unavailable.

## Baseline Interpretation

The stored `examples/results` files are workbook-oriented result sets containing
additional total-process and total-site targets. They are not exact snapshots
of the default `pinch_analysis_service` call and therefore are not used as a
byte-for-byte main-contract oracle. Exact numerical regression remains covered
by the representative main-contract case and the specialist domain, HPR, and
HEN fixture suites.
