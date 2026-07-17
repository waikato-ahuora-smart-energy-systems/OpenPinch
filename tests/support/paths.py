"""Stable repository and fixture paths for tests at any package depth."""

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TESTS_ROOT.parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"

__all__ = ["FIXTURES_ROOT", "REPOSITORY_ROOT", "TESTS_ROOT"]
