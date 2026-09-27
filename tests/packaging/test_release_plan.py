"""Release planning decisions made on every push to main."""

import pytest

from scripts.release_plan import UnsafeRelease, decide, main

HEAD = "a" * 40
OTHER = "b" * 40


def files(version):
    return [f"openpinch-{version}-py3-none-any.whl", f"openpinch-{version}.tar.gz"]


def plan(version, tags=(), releases=(), tag_sha=""):
    return decide(
        version=version,
        tags=list(tags),
        releases=list(releases),
        tag_sha=tag_sha,
        head_sha=HEAD,
    )


def test_new_version_above_every_tag_publishes():
    assert plan("0.6.11", tags=["v0.6.10", "v0.5.9"])[0] is True


def test_complete_published_release_is_skipped_even_with_extra_assets():
    release = {"tag": "v0.6.10", "assets": [*files("0.6.10"), "release-manifest.json"]}
    publish, reason = plan("0.6.10", ["v0.6.10"], [release], tag_sha=OTHER)
    assert publish is False
    assert "already released" in reason


def test_tagged_release_missing_an_asset_is_resumed():
    release = {"tag": "v0.6.11", "assets": files("0.6.11")[:1]}
    publish, reason = plan("0.6.11", ["v0.6.10", "v0.6.11"], [release], tag_sha=HEAD)
    assert publish is True
    assert "Resuming" in reason


def test_tag_without_release_on_this_commit_is_resumed():
    assert plan("0.6.11", ["v0.6.11"], tag_sha=HEAD)[0] is True


def test_stale_lower_version_is_blocked():
    with pytest.raises(UnsafeRelease, match="lower than the existing tag v0.6.12"):
        plan("0.6.11", tags=["v0.6.10", "v0.6.12"])


def test_semver_not_string_ordering():
    assert plan("0.10.0", tags=["v0.9.9"])[0] is True
    with pytest.raises(UnsafeRelease):
        plan("0.9.10", tags=["v0.10.0"])


def test_existing_tag_on_another_commit_is_blocked():
    with pytest.raises(UnsafeRelease, match="already points to"):
        plan("0.6.11", tags=["v0.6.11"], tag_sha=OTHER)


def test_non_release_tags_are_ignored():
    assert plan("0.6.11", tags=["v0.6.10", "v1.0.0rc1", "latest"])[0] is True


def test_cli_writes_github_output(tmp_path, capsys):
    tags = tmp_path / "tags.txt"
    tags.write_text("refs/tags/v0.6.10\n")
    releases = tmp_path / "releases.jsonl"
    releases.write_text(
        '{"tag": "v0.6.10", "assets": ["openpinch-0.6.10-py3-none-any.whl", '
        '"openpinch-0.6.10.tar.gz"]}\n'
    )
    args = ["--tags-file", str(tags), "--releases-file", str(releases)]
    assert main(["--version", "0.6.11", "--head-sha", HEAD, *args]) == 0
    assert capsys.readouterr().out == "publish=true\n"
    assert main(["--version", "0.6.9", "--head-sha", HEAD, *args]) == 1
