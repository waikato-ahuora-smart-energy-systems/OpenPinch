from __future__ import annotations

from pathlib import Path

from OpenPinch.resources import copy_sample_case, list_sample_cases


def test_copy_sample_case_uses_original_name_for_directory_destinations(
    tmp_path: Path,
):
    sample_name = list_sample_cases()[0]
    destination = tmp_path / "samples"
    destination.mkdir()

    copied = copy_sample_case(sample_name, destination)

    assert copied == destination / sample_name
    assert copied.exists()
