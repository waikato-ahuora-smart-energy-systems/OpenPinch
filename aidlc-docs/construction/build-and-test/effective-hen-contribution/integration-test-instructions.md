# Integration verification

Run `.venv/bin/pytest -q tests/analysis/heat_exchanger_networks`.
The run produced 459 passed, 1 skipped and 2 failures. The stale adapter test
was corrected to configure a real zone multiplier; all 28 adapter tests then
passed. The image-export test failed because Chrome could not start in the
sandbox, and passed when rerun outside it. No remaining observed failures.

Final PDM follow-up full-suite run: 463 passed, 1 skipped in 113.37 seconds. Run outside the sandbox when required for Chrome image export.
