"""Compatibility wrapper for invoking the packaged OpenPinch CLI."""

from OpenPinch.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
