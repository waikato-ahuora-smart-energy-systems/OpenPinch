# Technology Stack

## Programming Languages and Formats

- Python 3.14.2 or newer - package, tests, scripts, dashboard, and documentation configuration.
- reStructuredText - Sphinx user and API documentation.
- Markdown - repository guides, reports, and workflow documentation.
- TOML - package, build, dependency, formatter, and version-bump configuration.
- YAML - GitHub Actions and Read the Docs configuration.
- JSON - sample cases, schemas at runtime, solver summaries, manifests, and workspace bundles.
- XLSB/XLSX and CSV - legacy studies, input templates, fixtures, and exports.
- Jupyter notebooks - packaged learning and workflow examples.

## Runtime Frameworks and Libraries

- NumPy `<3` - vectorized numerical data and solver arrays.
- pandas `<3` - tables, summaries, comparisons, and exports.
- Pint `<1` - dimensional values and unit conversion.
- CoolProp `<8` - fluid properties and thermodynamic states.
- Pydantic `<3` - typed schemas, validation, and serialization.
- SciPy `<2` - interpolation and optimization.

## Optional Application Libraries

- Streamlit - interactive result dashboard.
- Plotly and Kaleido - interactive graphing and static graph export.
- openpyxl and pyxlsb - Excel export and legacy workbook input.
- TESPy - Brayton-cycle thermodynamics.
- Pyomo, GEKKO, and IDAES-PSE - HEN optimization models and solver integration.
- wakepy - prevent sleep during long synthesis execution.
- IPython kernel and nbformat - notebook execution and packaging checks.

## Build and Packaging

- Hatchling - PEP 517 wheel and source-distribution backend.
- uv and `uv.lock` - local environment and dependency lock management.
- `build` - distribution frontend used by `scripts/build_dist.py`.
- `bump-my-version` - automated version changes in pull-request automation.
- PyPI trusted publishing - TestPyPI and production PyPI release path.

## Testing and Quality

- pytest - 1,905 collected tests at analysis time, with synthesis and solver markers.
- Coverage.py - branch-independent statement coverage, enforced at 95% in CI; observed 99% locally.
- Ruff - linting, import ordering, bugbear mutable-default checks, and formatting support.
- Black - configured as an additional developer formatter.
- Pylint - installed in the development group but not configured or invoked in CI.
- Sphinx 9 with Read the Docs theme - documentation and API generation.

## Infrastructure and Delivery

- GitHub Actions - tests, docs, optional-install smoke tests, artifact builds, cross-platform wheel smoke tests, version bumps, and releases.
- Read the Docs - hosted documentation.
- PyPI - package distribution.
- No runtime cloud infrastructure, database, web framework, or container image.

