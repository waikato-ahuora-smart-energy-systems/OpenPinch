"""Generated and example checks for complete lane evidence."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.ci_policy import SELECTORS, gate_errors, required_jobs


@given(st.sampled_from(["integration", "full"]), st.data())
def test_gate_matches_reference_predicate(profile, data):
    names = sorted(required_jobs(profile))
    conclusions = data.draw(
        st.lists(
            st.sampled_from(["success", "failure", "skipped", "cancelled"]),
            min_size=len(names),
            max_size=len(names),
        )
    )
    results = [
        dict(name=n, conclusion=c) for n, c in zip(names, conclusions, strict=True)
    ]
    assert (not gate_errors(results, profile)) == all(
        c == "success" for c in conclusions
    )
    assert gate_errors(list(reversed(results)), profile) == gate_errors(
        results, profile
    )


@given(st.sampled_from(sorted(required_jobs("full"))))
def test_missing_or_duplicate_evidence_never_passes(name):
    results = [dict(name=n, conclusion="success") for n in required_jobs("full")]
    assert not gate_errors(results, "full")
    assert gate_errors([r for r in results if r["name"] != name], "full")
    assert gate_errors(results + [dict(name=name, conclusion="success")], "full")


def test_profiles_preserve_specialized_lanes():
    assert required_jobs("full") - required_jobs("integration") == {"solver-tests"}
    assert "performance-tests" in required_jobs("integration")
    assert len(SELECTORS) == 5
    with pytest.raises(ValueError):
        required_jobs("unknown")
