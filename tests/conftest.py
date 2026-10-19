"""Hypothesis settings shared by every environment the suite runs in."""

from hypothesis import settings

# Hypothesis loads its own "ci" profile (derandomize=True) when CI is set, so
# --hypothesis-seed would be ignored on GitHub but honoured locally and in
# Pyodide. This profile makes the seed choose the examples everywhere.
# deadline=None: timing depends on the machine, and WebAssembly runs several
# times slower than native CPython. database=None: examples saved by earlier
# local runs would otherwise change what a seeded run tries.
settings.register_profile(
    "openpinch", derandomize=False, deadline=None, database=None, print_blob=True
)
settings.load_profile("openpinch")
