// Run part of the pytest suite inside Pyodide (CPython on WebAssembly).
//
// usage: node scripts/pyodide/run_tests.mjs <openpinch wheel> [pytest args...]
//
// Node runs the same WebAssembly runtime as the browser, so this checks the
// browser engine of the OpenPinch web app. The built wheel is installed
// without dependencies: Pyodide ships its own numpy, scipy, pandas and
// pydantic builds, and its CoolProp 7.2 stands in for CoolProp 8 (the app
// uses a CoolProp 8 JS build behind a shim); loadPackage downloads them from
// the jsDelivr CDN for the pinned Pyodide version. pytest, hypothesis and the
// optional plotly, pyxlsb, tespy and gekko come from PyPI as pure-Python
// wheels at the uv.lock versions pinned in requirements.txt. The checkout is
// mounted only for its tests; the package under test is the installed wheel.
import { readFileSync } from 'node:fs'
import { basename, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadPyodide } from 'pyodide'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..', '..')
const requirements = readFileSync(resolve(HERE, 'requirements.txt'), 'utf8')
  .split('\n')
  .map((line) => line.split('#')[0].trim())
  .filter(Boolean)
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
pyodide.globals.set('REQUIREMENTS', pyodide.toPy(requirements))
const exitCode = await pyodide.runPythonAsync(`
import os, sys, micropip
# The constraints hold transitive dependencies to their pins as well.
await micropip.install(list(REQUIREMENTS), constraints=list(REQUIREMENTS))
pinned = {r.split("==")[0].lower() for r in REQUIREMENTS}
unpinned = sorted(
    f"{name}=={pkg.version}"
    for name, pkg in micropip.list().items()
    if pkg.source == "pypi" and name.lower() not in pinned
)
if unpinned:
    raise SystemExit(f"Pin these in scripts/pyodide/requirements.txt: {unpinned}")
await micropip.install("emfs:/tmp/${wheelName}", deps=False)
os.chdir("/work")
# Import the installed wheel, not the mounted checkout: the checkout goes
# last, after site-packages, so only the tests package (shared strategies)
# comes from it. pytest.ini's "pythonpath = ." would put it first, so that
# setting is cleared.
sys.path = [p for p in sys.path if p not in ("", "/work")] + ["/work"]
import pytest
int(pytest.main(["-p", "no:cacheprovider", "--import-mode=importlib", "-o", "pythonpath=", *PYTEST_ARGS]))
`)
process.exit(Number(exitCode))
