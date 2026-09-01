"""Reusable path cases for installed-artifact boundary checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hypothesis import strategies as st


@dataclass(frozen=True)
class ArtifactImportPathCase:
    """One source-tree or checkout-local environment import location."""

    checkout_name: str
    environment_parts: tuple[str, ...]
    site_packages_parts: tuple[str, ...]
    source_import: bool

    @property
    def repo_root(self) -> Path:
        """Return a generated absolute checkout path."""
        return Path("/generated-checkouts") / self.checkout_name

    @property
    def package_path(self) -> Path:
        """Return the generated package ``__init__`` path."""
        if self.source_import:
            return self.repo_root / "OpenPinch" / "__init__.py"
        return (
            self.repo_root
            / ".environments"
            / Path(*self.environment_parts)
            / Path(*self.site_packages_parts)
            / "OpenPinch"
            / "__init__.py"
        )


_safe_part = st.from_regex(r"[a-z][a-z0-9_-]{0,11}", fullmatch=True)


@st.composite
def artifact_import_path_cases(draw):
    """Generate structured source and checkout-local environment imports."""
    return ArtifactImportPathCase(
        checkout_name=draw(_safe_part),
        environment_parts=tuple(draw(st.lists(_safe_part, min_size=1, max_size=3))),
        site_packages_parts=draw(
            st.sampled_from(
                (
                    ("lib", "python3.14", "site-packages"),
                    ("Lib", "site-packages"),
                )
            )
        ),
        source_import=draw(st.booleans()),
    )


__all__ = ["ArtifactImportPathCase", "artifact_import_path_cases"]
