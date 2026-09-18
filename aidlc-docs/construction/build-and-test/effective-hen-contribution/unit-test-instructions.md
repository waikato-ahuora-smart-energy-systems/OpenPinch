# Unit verification

Run `.venv/bin/pytest -q tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_segmented_streams.py`.
Result: 102 passed, including fixed-seed Hypothesis, tolerance boundaries,
multiplier use, period selection and no additional dTmin scaling or floor.

PDM follow-up: include `tests/analysis/heat_exchanger_networks/test_heat_exchanger_network_pinch_parity.py`; the three-module run passed 113 tests.
