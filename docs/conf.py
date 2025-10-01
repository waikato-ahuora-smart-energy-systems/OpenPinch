"""Sphinx configuration for the OpenPinch documentation site."""

from __future__ import annotations

import os
import sys
from datetime import datetime

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - safety net for older interpreters
    import tomli as tomllib  # type: ignore

# -- Path setup --------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -- Project information -----------------------------------------------------
project = "OpenPinch"
author = "Tim Walmsley"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"


def _read_version(default: str = "0.0.0") -> str:
    """Pull the package version from ``pyproject.toml`` for consistent docs."""
    pyproject_path = os.path.join(PROJECT_ROOT, "pyproject.toml")
    try:
        with open(pyproject_path, "rb") as fh:
            data = tomllib.load(fh)
        return data["project"]["version"]
    except Exception:
        return default


release = _read_version()
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autoclass_content = "both"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = [
    "CoolProp",
    "coolprop",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "pint",
    "pyxlsb",
    "scipy",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
try:  # Prefer the Read the Docs theme when available
    import sphinx_rtd_theme  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fall back for local builds
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
