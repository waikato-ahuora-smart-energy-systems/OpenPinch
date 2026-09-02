"""Tests for retry-safe package-index release verification."""

from __future__ import annotations

import hashlib
import io
import json
import runpy
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tests.support.paths import REPOSITORY_ROOT

SCRIPT = REPOSITORY_ROOT / "scripts" / "check_package_index_release.py"


def _load_namespace() -> dict:
    return runpy.run_path(str(SCRIPT))


def _write_distributions(dist_dir: Path, version: str = "1.2.3") -> dict[str, str]:
    dist_dir.mkdir()
    files = {
        f"openpinch-{version}-py3-none-any.whl": b"wheel-content",
        f"openpinch-{version}.tar.gz": b"sdist-content",
    }
    for name, content in files.items():
        (dist_dir / name).write_bytes(content)
    return {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }


def _response(files: dict[str, str]):
    payload = {
        "urls": [
            {"filename": name, "digests": {"sha256": digest}}
            for name, digest in files.items()
        ]
    }
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_inspection_reports_absent_release_for_http_404(tmp_path):
    namespace = _load_namespace()
    expected = _write_distributions(tmp_path / "dist")

    def missing(_request, timeout):
        assert timeout > 0
        raise HTTPError("https://index.invalid", 404, "missing", {}, None)

    result = namespace["inspect_release"](
        index_url="https://index.invalid/pypi",
        project="OpenPinch",
        version="1.2.3",
        expected_files=expected,
        opener=missing,
    )

    assert result == "absent"


@pytest.mark.parametrize("status_code", [429, 503])
def test_inspection_retries_transient_index_responses(tmp_path, status_code):
    namespace = _load_namespace()
    expected = _write_distributions(tmp_path / "dist")
    calls = 0

    def transient_then_complete(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError("https://index.invalid", status_code, "transient", {}, None)
        return _response(expected)

    result = namespace["inspect_release"](
        index_url="https://index.invalid/pypi",
        project="OpenPinch",
        version="1.2.3",
        expected_files=expected,
        opener=transient_then_complete,
        attempts=2,
        retry_delay=0,
    )

    assert result == "complete"
    assert calls == 2


@pytest.mark.parametrize(
    "file_count, expected_status", [(1, "partial"), (2, "complete")]
)
def test_inspection_accepts_only_matching_partial_or_complete_files(
    tmp_path, file_count, expected_status
):
    namespace = _load_namespace()
    expected = _write_distributions(tmp_path / "dist")
    published = dict(list(expected.items())[:file_count])

    result = namespace["inspect_release"](
        index_url="https://index.invalid/pypi/",
        project="OpenPinch",
        version="1.2.3",
        expected_files=expected,
        opener=lambda _request, timeout: _response(published),
    )

    assert result == expected_status


@pytest.mark.parametrize(
    "published",
    [
        {"openpinch-1.2.3-py3-none-any.whl": "0" * 64},
        {"unexpected-1.2.3.tar.gz": "1" * 64},
    ],
)
def test_inspection_rejects_mismatched_or_unexpected_files(tmp_path, published):
    namespace = _load_namespace()
    expected = _write_distributions(tmp_path / "dist")

    with pytest.raises(namespace["ReleaseValidationError"]):
        namespace["inspect_release"](
            index_url="https://index.invalid/pypi",
            project="OpenPinch",
            version="1.2.3",
            expected_files=expected,
            opener=lambda _request, timeout: _response(published),
        )


def test_cli_allows_exact_partial_preflight_but_requires_complete_postflight(
    tmp_path, monkeypatch, capsys
):
    namespace = _load_namespace()
    expected = _write_distributions(tmp_path / "dist")
    partial = dict(list(expected.items())[:1])
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "urlopen",
        lambda _request, timeout: _response(partial),
    )
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "wait_for_complete_release",
        lambda *, inspect: inspect(),
    )

    common = [
        "--index-url",
        "https://index.invalid/pypi",
        "--project",
        "OpenPinch",
        "--version",
        "1.2.3",
        "--dist-dir",
        str(tmp_path / "dist"),
    ]
    assert namespace["main"]([*common, "--allow-partial"]) == 0
    assert capsys.readouterr().out.strip() == "partial"
    assert namespace["main"]([*common, "--require-complete"]) == 1


def test_postflight_retries_absent_and_partial_states_until_complete(tmp_path):
    namespace = _load_namespace()
    _write_distributions(tmp_path / "dist")
    states = iter(["absent", "partial", "complete"])
    sleeps: list[float] = []

    status = namespace["wait_for_complete_release"](
        inspect=lambda: next(states),
        attempts=5,
        retry_delay=10.0,
        sleeper=sleeps.append,
    )

    assert status == "complete"
    assert sleeps == [10.0, 10.0]
