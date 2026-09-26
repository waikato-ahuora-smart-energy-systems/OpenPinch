"""Metadata edits never trigger costly validation; base edits do."""

import pytest

from scripts.plan_ci_event import plan


@pytest.mark.parametrize("base", ["main", "develop"])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize(
    "action,changes,run",
    [
        ("synchronize", {}, True),
        ("edited", {"body": {}}, False),
        ("edited", {"base": {}}, True),
        ("ready_for_review", {}, True),
    ],
)
def test_event_plan(base, draft, action, changes, run):
    result = plan(
        {
            "action": action,
            "changes": changes,
            "pull_request": {"draft": draft, "base": {"ref": base}},
        }
    )
    assert result == {
        "profile": "full" if base == "main" else "integration",
        "run": str(run and not draft).lower(),
    }
