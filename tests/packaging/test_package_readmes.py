"""The directory READMEs are contracts, so what they cite must exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
READMES = sorted(
    path
    for folder in ("OpenPinch", "scripts")
    for path in (ROOT / folder).rglob("README.md")
    if "node_modules" not in path.parts
)
LINK = re.compile(r"\]\((?!https?:)([^)#]+)\)")
CITATION = re.compile(r"\[([\w./-]+\.(?:py|sh|mjs)(?::[\w.]+)?(?:, [\w./:-]+)*)\]")
TEST_PATH = re.compile(r"(?<![\w/])tests/[\w./-]+\.py")


def _cited_files(readme: Path, name: str) -> list[Path]:
    """A citation is relative to the README, the repository or the package."""
    candidates = [readme.parent / name, ROOT / name, ROOT / "OpenPinch" / name]
    if not any(path.exists() for path in candidates):
        candidates = list((ROOT / "OpenPinch").rglob(name))
    return [path for path in candidates if path.exists()]


def _citations(text: str) -> list[tuple[str, str]]:
    """A bracket cites file:symbol items; a bare symbol uses the previous file."""
    citations = []
    for group in CITATION.findall(text):
        name = ""
        for item in group.split(", "):
            cited, separator, symbol = item.partition(":")
            if separator or re.search(r"\.(?:py|sh|mjs)$", cited):
                name = cited
            else:
                symbol = cited
            citations.append((name, symbol))
    return citations


def _broken_references(readme: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    broken = [
        f"link {target}"
        for target in LINK.findall(text)
        if not (readme.parent / target).exists()
    ]
    broken += [
        f"test path {path}"
        for path in TEST_PATH.findall(text)
        if not (ROOT / path).exists()
    ]
    for name, symbol in _citations(text):
        files = _cited_files(readme, name)
        if not files:
            broken.append(f"file {name}")
        elif symbol and not any(
            re.search(
                rf"\b{re.escape(symbol.split('.')[-1])}\b", path.read_text("utf-8")
            )
            for path in files
        ):
            broken.append(f"symbol {name}:{symbol}")
    return broken


def test_package_has_directory_readmes():
    assert ROOT / "OpenPinch" / "README.md" in READMES


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(ROOT)))
def test_readme_references_exist(readme):
    assert _broken_references(readme) == []


def test_grouped_citations_are_each_checked(tmp_path):
    (tmp_path / "module.py").write_text("def present():\n    pass\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "- A rule [module.py:present, absent_a] and another "
        "[module.py:present, module.py:absent_b].\n",
        encoding="utf-8",
    )

    assert _broken_references(readme) == [
        "symbol module.py:absent_a",
        "symbol module.py:absent_b",
    ]
