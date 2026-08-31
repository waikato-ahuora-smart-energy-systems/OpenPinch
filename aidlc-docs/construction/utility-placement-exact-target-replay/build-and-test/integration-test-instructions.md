# Integration Test Instructions

Run the public hierarchy, notebook, and documentation gates:

```bash
uv run pytest -q tests/application/test_utility_placement.py
uv run pytest -q tests/packaging/test_notebooks.py tests/packaging/test_docs_consistency.py
```

Execute notebook 19 with the repository Python kernel. Inspect its Process and
Site comparison tables: ordinary retargeted duties must reproduce optimizer
evidence, and the standard GCC and Total Site Profile must render. The notebook
is a demonstration and must contain no tests or assertions.

The application suite covers Process, Site, Community, Region, multiperiod,
capacity-limited, source-isolation, and returned-case replay behavior.
