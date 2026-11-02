"""Check that the packages only the synthesis extra asks for are not installed."""

from __future__ import annotations

import sys
from importlib.util import find_spec

STACK_MODULES = ("pyomo", "gekko", "kaleido", "wakepy", "idaes")


def main() -> int:
    found = [module for module in STACK_MODULES if find_spec(module) is not None]
    if found:
        print(f"Synthesis stack is installed: {', '.join(found)}", file=sys.stderr)
        return 1
    print("Synthesis stack is absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
