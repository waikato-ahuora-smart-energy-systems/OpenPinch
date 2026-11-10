"""The generated configuration options reference matches the live metadata."""

from __future__ import annotations

from OpenPinch.presentation.configuration import configuration_options
from tests.support.paths import REPOSITORY_ROOT


def test_committed_configuration_reference_matches_the_generator():
    from scripts.generate_configuration_reference import (
        DESTINATION,
        render_reference,
    )

    assert DESTINATION.read_text(encoding="utf-8") == render_reference()


def test_configuration_reference_lists_every_option_with_its_default():
    from scripts.generate_configuration_reference import render_reference

    text = render_reference()
    for field in configuration_options():
        assert f"``{field.name}``" in text
    assert '"``HPR_AMBIENT_AIR_ENABLED``","boolean","``True``"' in text
    assert '"``ORC_ETA_TURBINE``","number","``0.8``","","> 0 to 1"' in text


def test_configuration_reference_page_is_in_the_api_navigation():
    docs = REPOSITORY_ROOT / "docs"
    page = (docs / "api" / "configuration-options.rst").read_text(encoding="utf-8")
    index = (docs / "api" / "index.rst").read_text(encoding="utf-8")

    assert ".. include:: ../_data/configuration-options.inc" in page
    assert "   configuration-options\n" in index
