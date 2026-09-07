# Cleanup build verification

Use the existing development environment from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m build --no-isolation --outdir /tmp/openpinch-cleanup-build
```

Wheel and sdist 0.6.5 built successfully. Wheel inspection confirmed all 19 maintained tutorials and the package entrypoint. Output stays outside the cleaned workspace.
