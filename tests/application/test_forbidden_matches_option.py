"""``HENS_FORBIDDEN_MATCHES``: matches a design may not use.

No option forbade a match. Each entry is a pair of names, the hot side then
the cold side; either may be a process stream or a utility, so
``["H1", "C2"]`` forbids that recovery match and ``["HPS", "C1"]`` forbids
heating C1 with HPS.
"""

from __future__ import annotations

import pytest

from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS


def test_forbidden_matches_are_a_configuration_key_of_name_pairs():
    spec = USER_CONFIG_FIELD_SPECS["HENS_FORBIDDEN_MATCHES"]

    assert spec.default == []
    assert spec.config_path == ("hens", "forbidden_matches")
    assert Configuration().hens.forbidden_matches == []
    configured = Configuration(
        options={"HENS_FORBIDDEN_MATCHES": [["H1", "C2"], (" HPS ", "C1")]}
    )
    assert configured.hens.forbidden_matches == [["H1", "C2"], ["HPS", "C1"]]
    for bad in ("H1->C2", [["H1"]], [["H1", "C2", "C3"]], [["H1", " "]], ["H1", "C2"]):
        with pytest.raises(ValueError, match="HENS_FORBIDDEN_MATCHES"):
            Configuration(options={"HENS_FORBIDDEN_MATCHES": bad})
