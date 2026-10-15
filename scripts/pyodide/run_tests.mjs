// Run part of the pytest suite inside Pyodide (CPython on WebAssembly).
//
// usage: node scripts/pyodide/run_tests.mjs <openpinch wheel> [pytest args...]
//
// Node runs the same WebAssembly runtime as the browser, so this checks the
// browser engine of the OpenPinch web app. The built wheel is installed
// without dependencies: Pyodide ships its own numpy, scipy, pandas and
// pydantic builds, and its CoolProp 7.2 stands in for CoolProp 8 (the app
// uses a CoolProp 8 JS build behind a shim). The checkout is mounted only for
// its tests; the package under test is the installed wheel.
import { readFileSync } from 'node:fs'
import { basename, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadPyodide } from 'pyodide'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const [wheel, ...pytestArgs] = process.argv.slice(2)
if (!wheel) {
  console.error('usage: node scripts/pyodide/run_tests.mjs <wheel> [pytest args...]')
  process.exit(2)
}

const pyodide = await loadPyodide()
await pyodide.loadPackage(['micropip', 'numpy', 'scipy', 'pandas', 'pydantic', 'coolprop', 'pyyaml'])

const wheelName = basename(wheel)
pyodide.FS.writeFile(`/tmp/${wheelName}`, readFileSync(wheel))
pyodide.FS.mkdirTree('/work')
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: REPO }, '/work')

pyodide.globals.set('PYTEST_ARGS', pyodide.toPy(pytestArgs))
const exitCode = await pyodide.runPythonAsync(`
import os, sys, micropip
await micropip.install(["pint>=0.24.4", "pytest", "hypothesis", "jsonschema", "openpyxl"])
await micropip.install("emfs:/tmp/${wheelName}", deps=False)
os.chdir("/work")
# Import the installed wheel, not the mounted checkout.
sys.path = [p for p in sys.path if p not in ("", "/work")]
import pytest
int(pytest.main(["-p", "no:cacheprovider", "--import-mode=importlib", *PYTEST_ARGS]))
`)
process.exit(Number(exitCode))
