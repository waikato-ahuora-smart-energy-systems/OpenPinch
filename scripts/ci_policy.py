"""Single delivery-lane policy and fail-closed decisions (no network access)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

POLICY = "delivery-v1"
SURFACES = ("core", "dashboard", "notebook", "brayton_cycle", "tespy", "synthesis")
RUNNERS = ("ubuntu-latest", "windows-latest", "macos-latest")
SELECTORS = {
    "test": "not solver and not tespy and not performance and not docs",
    "docs": "docs",
    "hpr-tespy-tests": "tespy",
    "performance-tests": "performance",
    "solver-tests": "solver",
}
SHARED_JOBS = frozenset(
    {
        "test",
        "docs",
        "hpr-tespy-tests",
        "performance-tests",
        "artifact-build",
        "artifact-install-tespy-smoke",
        "workflow-lint",
    }
)


def required_jobs(profile: str, *, expanded: bool = True) -> frozenset[str]:
    """Return every required lane, optionally expanding smoke matrices."""
    if profile not in {"integration", "full"}:
        raise ValueError("Unknown validation profile")
    matrices = (
        {f"optional-install-smoke ({s})" for s in SURFACES}
        | {f"artifact-install-smoke ({r})" for r in RUNNERS}
        if expanded
        else {"optional-install-smoke", "artifact-install-smoke"}
    )
    return SHARED_JOBS | matrices | ({"solver-tests"} if profile == "full" else set())


def gate_errors(
    results: list[dict], profile: str, *, expanded: bool = True
) -> list[str]:
    """Reject missing, duplicate, skipped, and unsuccessful mandatory evidence."""
    errors = []
    for name in sorted(required_jobs(profile, expanded=expanded)):
        matches = [item for item in results if item.get("name") == name]
        if len(matches) != 1 or matches[0].get("conclusion") != "success":
            errors.append(f"{name}: expected exactly one successful result")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "selector", "gate"])
    parser.add_argument(
        "--profile", default="integration", choices=["integration", "full"]
    )
    parser.add_argument("--lane", choices=SELECTORS)
    args = parser.parse_args()
    if args.command == "matrix":
        print(f"surfaces={json.dumps(SURFACES)}")
        print(f"runners={json.dumps(RUNNERS)}")
    elif args.command == "selector":
        if not args.lane:
            parser.error("selector requires --lane")
        print(SELECTORS[args.lane])
    else:
        needs = json.loads(os.environ["LANE_RESULTS"])
        results = [
            {"name": name, "conclusion": data["result"]} for name, data in needs.items()
        ]
        if os.environ.get("REUSE_INTEGRATION") == "true":
            # Only caller-verified integration proof can replace skipped lanes.
            for item in results:
                if (
                    item["name"] in required_jobs("integration", expanded=False)
                    and item["conclusion"] == "skipped"
                ):
                    item["conclusion"] = "success"
        errors = gate_errors(results, args.profile, expanded=False)
        summary = (
            f"Policy: {POLICY}; profile: {args.profile}; "
            f"integration reuse: {os.environ.get('REUSE_INTEGRATION', 'false')}\n"
            + ("\n".join(errors) if errors else "All mandatory lanes passed.")
        )
        print(summary)
        if path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(path).open("a") as handle:
                handle.write(summary + "\n")
        return int(bool(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
