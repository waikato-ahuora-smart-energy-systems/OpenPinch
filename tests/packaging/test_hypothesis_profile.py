"""Hypothesis runs the same settings on GitHub, locally and in Pyodide."""

from hypothesis import settings


def test_the_openpinch_profile_is_loaded():
    # Hypothesis loads its own "ci" profile (derandomize=True) when CI is set,
    # which would make --hypothesis-seed inert on the GitHub lanes only.
    assert settings.get_current_profile_name() == "openpinch"
    assert settings.default.derandomize is False
    assert settings.default.deadline is None
    assert settings.default.database is None
