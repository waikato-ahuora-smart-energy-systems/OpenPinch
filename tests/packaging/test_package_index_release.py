"""Tests for retry-safe package-index release verification."""

from __future__ import annotations

import hashlib
import io
import json
import runpy
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from hypothesis import given
from hypothesis import strategies as st

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

    now = [0.0]
    result = namespace["wait_for_complete_release"](
        inspect=lambda timeout: namespace["inspect_release"](
            index_url="https://index.invalid/pypi",
            project="OpenPinch",
            version="1.2.3",
            expected_files=expected,
            opener=transient_then_complete,
            timeout=timeout,
        ),
        clock=lambda: now[0],
        sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
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
        lambda *, inspect, **kwargs: inspect(1),
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
        inspect=lambda timeout: next(states),
        budget=50,
        retry_delay=10.0,
        sleeper=sleeps.append,
    )

    assert status == "complete"
    assert sleeps == [10.0, 10.0]


@pytest.mark.parametrize("complete_at", [60, 120, 290])
def test_visibility_beyond_old_fifty_second_window(complete_at):
    namespace = _load_namespace()
    now = [0.0]
    assert (
        namespace["wait_for_complete_release"](
            inspect=lambda timeout: "complete" if now[0] >= complete_at else "absent",
            clock=lambda: now[0],
            sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
        )
        == "complete"
    )
    assert now[0] == complete_at


def test_deadline_counts_network_time_and_rejects_late_success():
    namespace = _load_namespace()
    now = [0.0]

    def late(timeout):
        assert timeout == 5
        now[0] += 6
        return "complete"

    with pytest.raises(namespace["ReleaseValidationError"], match="deadline"):
        namespace["wait_for_complete_release"](
            inspect=late, budget=5, clock=lambda: now[0]
        )


@pytest.mark.parametrize("budget", [0, -1, 601, float("inf"), float("nan")])
def test_invalid_budget_never_calls_transport(budget):
    namespace = _load_namespace()
    with pytest.raises(ValueError):
        namespace["wait_for_complete_release"](
            inspect=lambda timeout: pytest.fail("transport"), budget=budget
        )


def test_retry_after_cannot_extend_deadline():
    namespace = _load_namespace()
    now = [0.0]
    calls = []

    def unavailable(timeout):
        calls.append(timeout)
        raise namespace["TransientIndexError"]("limited", retry_after=500)

    with pytest.raises(
        namespace["ReleaseValidationError"], match="transport exhaustion"
    ):
        namespace["wait_for_complete_release"](
            inspect=unavailable,
            budget=30,
            clock=lambda: now[0],
            sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
        )
    assert calls == [20]
    assert now[0] == 30


@pytest.mark.parametrize("body", [b"[]", b"broken", b"x" * 2_000_001])
def test_bad_response_is_permanent(body):
    namespace = _load_namespace()
    with pytest.raises(namespace["ReleaseValidationError"]):
        namespace["inspect_release"](
            index_url="https://index.invalid",
            project="OpenPinch",
            version="1.2.3",
            expected_files={},
            opener=lambda *a, **k: io.BytesIO(body),
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://index.invalid",
        "https://user:secret@index.invalid",
        "https://index.invalid?token=secret",
    ],
)
def test_unsafe_index_url_is_rejected_without_disclosing_it(url):
    namespace = _load_namespace()
    with pytest.raises(namespace["ReleaseValidationError"]) as error:
        namespace["inspect_release"](
            index_url=url,
            project="OpenPinch",
            version="1",
            expected_files={},
            opener=lambda *a, **k: pytest.fail("transport"),
        )
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("https://index.invalid", 401, "denied", {}, None),
        HTTPError("https://index.invalid", 403, "denied", {}, None),
        ssl.SSLCertVerificationError("certificate"),
        URLError(ssl.SSLCertVerificationError("certificate")),
    ],
)
def test_auth_and_tls_failures_never_retry(failure):
    from scripts import check_package_index_release as checker

    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        raise failure

    with pytest.raises(checker.ReleaseValidationError) as error:
        checker.wait_for_complete_release(
            inspect=lambda timeout: checker.inspect_release(
                index_url="https://index.invalid",
                project="OpenPinch",
                version="1.2.3",
                expected_files={},
                opener=transport,
                timeout=timeout,
            ),
            sleeper=lambda _: pytest.fail("permanent failure must not sleep"),
        )
    assert not isinstance(error.value, checker.TransientIndexError)
    assert len(calls) == 1


def test_duplicate_index_files_fail_closed():
    from scripts import check_package_index_release as checker

    entry = {"filename": "wheel.whl", "digests": {"sha256": "a" * 64}}
    with pytest.raises(checker.ReleaseValidationError, match="repeats"):
        checker.inspect_release(
            index_url="https://index.invalid",
            project="OpenPinch",
            version="1.2.3",
            expected_files={"wheel.whl": "a" * 64},
            opener=lambda *a, **k: io.BytesIO(
                json.dumps({"urls": [entry, entry]}).encode()
            ),
        )


@given(
    st.integers(min_value=1, max_value=600), st.integers(min_value=1, max_value=1000)
)
def test_generated_retry_sequences_never_extend_budget(budget, retry_delay):
    from scripts import check_package_index_release as checker

    now = [0.0]
    calls = []

    def inspect(timeout):
        assert 0 < timeout <= min(20, budget - now[0])
        calls.append(timeout)
        return "partial"

    with pytest.raises(checker.ReleaseValidationError, match="deadline"):
        checker.wait_for_complete_release(
            inspect=inspect,
            budget=budget,
            retry_delay=retry_delay,
            clock=lambda: now[0],
            sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
        )
    assert now[0] == budget
    assert 1 <= len(calls) <= budget
