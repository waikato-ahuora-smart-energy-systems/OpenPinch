"""Resolve PR event policy without trusting branch names as candidate identity."""

import json
import os
from pathlib import Path


def plan(event: dict) -> dict[str, str]:
    pr = event["pull_request"]
    profile = "full" if pr["base"]["ref"] == "main" else "integration"
    metadata = event.get("action") == "edited" and "base" not in event.get(
        "changes", {}
    )
    return {"profile": profile, "run": str(not pr["draft"] and not metadata).lower()}


def main() -> None:
    result = plan(json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.write("".join(f"{key}={value}\n" for key, value in result.items()))


if __name__ == "__main__":
    main()
