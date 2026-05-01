"""Compatibility shim for running from a source checkout."""

import warnings

warnings.warn(
    "Import from 'OpenPinch' instead of the repository root package.",
    DeprecationWarning,
    stacklevel=2,
)

from OpenPinch import *  # noqa: F401,F403
