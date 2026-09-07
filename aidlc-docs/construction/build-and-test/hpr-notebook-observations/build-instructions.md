# Build instructions

From the repository root with the existing development environment:

```sh
.venv/bin/ruff check OpenPinch tests scripts
.venv/bin/python -m sphinx -W --keep-going -b html docs /tmp/openpinch-hpr-docs
.venv/bin/python -m build --no-isolation --outdir /tmp/openpinch-hpr-dist
git diff --check
```

Build the wheel from the sdist using the standard build frontend. Both artifacts
must contain the residual modules and canonical notebook 08. Install the wheel
without checkout imports and run scripts/artifact_install_smoke.py using the
matching dependency surface. Preserve all independently edited notebooks; do
not run the all-notebook writer against this working tree.
