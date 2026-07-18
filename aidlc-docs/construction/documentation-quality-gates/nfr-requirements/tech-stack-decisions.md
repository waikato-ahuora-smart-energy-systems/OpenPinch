# Tech Stack Decisions

Use the existing Python, pytest, Ruff, Sphinx, importlib-resources, and package
build toolchain. Use CSV as the shared CI/RTD boundary, AST for executable-cell
checks, and class introspection for live public inventory parity. No new runtime
dependency or infrastructure is introduced.
