import os
import sys


def main() -> None:
    print("sys.executable:", sys.executable)
    print("sys.version:", sys.version)
    print("cwd:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))

    try:
        import pytest  # noqa: F401

        print("pytest: import ok")
    except Exception as exc:  # pragma: no cover - debug probe
        print(f"pytest: import failed: {exc!r}")

    try:
        import OpenPinch  # noqa: F401

        print("OpenPinch: import ok")
    except Exception as exc:  # pragma: no cover - debug probe
        print(f"OpenPinch: import failed: {exc!r}")


if __name__ == "__main__":
    main()
