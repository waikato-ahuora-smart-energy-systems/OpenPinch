# Pyodide test runner

## Overview
Runs part of the pytest suite inside Pyodide (CPython on WebAssembly) under Node, to check the
engine the OpenPinch web app uses in the browser. It is a Node project, not part of the Python
distribution. The built OpenPinch wheel is installed without dependencies; the checkout is mounted
only to supply the tests. It is driven by the Pyodide lane of `.github/workflows/ci.yml`.

## Interface
- `run_tests.mjs` (`run_tests.mjs`): `node scripts/pyodide/run_tests.mjs <wheel> [pytest args...]`.
  Loads Pyodide, installs the pinned PyPI packages with micropip, installs the wheel, runs
  `pytest.main` in `/work` and exits with its exit code (2 if no wheel is given, 1 on an
  uncaught error).
- `requirements.txt` (`requirements.txt`): `name==version` pins for every package micropip
  fetches from PyPI, transitive ones included. Comments after `#` are ignored.
- `package.json` (`package.json`): declares the `pyodide` dependency (exact version) and
  `node >=22`; `package-lock.json` fixes it for `npm ci`.
- Subdirectories: none.

## Invariants
- Every package micropip takes from PyPI must be pinned in `requirements.txt`; otherwise the
  run raises `SystemExit` naming the missing pins [run_tests.mjs:runPythonAsync].
- Pins are installed with the same list as constraints, so transitive dependencies also stay at
  their pins [run_tests.mjs:runPythonAsync].
- The wheel is installed with `deps=False`; numpy, scipy, pandas, pydantic, coolprop and pyyaml
  come from Pyodide's own builds [run_tests.mjs:loadPackage].
- Tests import the installed wheel, not the checkout: `sys.path` drops `""` and `/work`, appends
  `/work` last, and pytest runs with `--import-mode=importlib` and `-o pythonpath=`
  [run_tests.mjs:runPythonAsync].
- Pins equal the `uv.lock` versions, checked by `tests/packaging/test_pyodide_pins.py`.
- Uncaught exceptions and rejections print only the error message, then exit 1
  [run_tests.mjs:reportAndExit].

## Side Effects
- Downloads Pyodide packages from the jsDelivr CDN and wheels from PyPI at run time
  [run_tests.mjs:loadPackage].
- Writes the wheel to `/tmp/<wheel name>` in Pyodide's virtual filesystem and mounts the
  repository root at `/work` through NODEFS (the checkout is visible to the tests, and pytest
  runs with `-p no:cacheprovider`) [run_tests.mjs].
- Reads `requirements.txt` and the wheel file from disk [run_tests.mjs].
- Calls `process.exit` and registers global `uncaughtException`/`unhandledRejection` handlers
  [run_tests.mjs].

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_pyodide_pins.py tests/packaging/test_pyodide_runner.py`
  (the runner test skips unless `npm ci` has been run in `scripts/pyodide`).
- Full lane: `uv build --out-dir dist` then `node scripts/pyodide/run_tests.mjs dist/*.whl -q tests/domain`.
