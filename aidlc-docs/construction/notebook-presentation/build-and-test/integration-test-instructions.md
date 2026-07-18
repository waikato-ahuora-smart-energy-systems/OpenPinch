# Notebook Presentation Integration Test Instructions

## Generator to Packaged Resource

1. Run the canonical generator twice.
2. Confirm both passes produce the same 18 notebook byte streams.
3. Confirm package resource discovery returns the coverage-manifest inventory.
4. Confirm every notebook has one review section, a following display cell,
   subject-specific interpretation, and source-only output state.

## Profile Execution

Run each optional profile with its declared environment:

```bash
OPENPINCH_TUTORIAL_PROFILES=slow-hpr \
  uv run pytest tests/packaging/test_notebooks.py::test_optional_profile_notebooks_execute -q

OPENPINCH_TUTORIAL_PROFILES=solver \
  uv run pytest tests/packaging/test_notebooks.py::test_optional_profile_notebooks_execute -q

OPENPINCH_TUTORIAL_PROFILES=interactive \
  uv run pytest tests/packaging/test_notebooks.py::test_optional_profile_notebooks_execute -q
```

The selected test must execute all notebooks assigned to that profile. Two
unselected parametrized profiles skip in each command.

## Distribution Integration

Build the distribution, install the wheel into a temporary site-packages
environment outside the checkout, and run:

```bash
/path/to/python scripts/artifact_install_smoke.py \
  --repo-root /path/to/OpenPinch
```

The smoke must import from the installed artifact, solve a public workflow,
construct a workspace, expose exactly the two root workflow classes, find the
packaged notebooks, reject retired packages, and run CLI notebook help.

## Cleanup

Delete only the explicit temporary documentation, distribution, and installed-
artifact directories after their evidence has been recorded. No repository
artifact requires cleanup.
