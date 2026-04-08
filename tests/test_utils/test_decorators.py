"""Regression tests for decorators utility helpers."""

import logging
import re
import time

from OpenPinch.lib import *
from OpenPinch.utils import *


def get_dummy_function():
    """Return a decorated dummy function used by this test module."""

    @timing_decorator
    def dummy(x, y):
        time.sleep(0.01)
        return x + y

    return dummy


def test_timing_decorator_enabled(monkeypatch):
    monkeypatch.setattr(config, "ACTIVATE_TIMING", True)
    monkeypatch.setattr(config, "LOG_TIMING", True)
    dummy = get_dummy_function()
    assert dummy(2, 3) == 5


def test_timing_decorator_disabled(monkeypatch):
    monkeypatch.setattr(config, "ACTIVATE_TIMING", False)
    monkeypatch.setattr(config, "LOG_TIMING", True)
    dummy = get_dummy_function()
    assert dummy(10, 20) == 30


def test_timing_decorator_logging_output(monkeypatch, caplog):
    monkeypatch.setattr(config, "ACTIVATE_TIMING", True)
    monkeypatch.setattr(config, "LOG_TIMING", True)
    caplog.set_level(logging.INFO)
    dummy = get_dummy_function()
    dummy(1, 2)

    logs = caplog.text
    # In your current decorator, logging is commented out — uncomment for this to work:
    # logger.info(f"Function '{func.__name__}' executed in {exec_time:.6f} seconds.")
    assert "Function 'dummy'" in logs
    assert re.search(r"executed in \d+\.\d{6} seconds", logs)


# ===== Merged from test_decorators_extra.py =====
"""Additional coverage tests for decorator helpers."""

import logging

from OpenPinch.utils import decorators


class _ClosedStream:
    closed = True


class _ClosedHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stream = _ClosedStream()
        self.was_closed = False

    def emit(self, record):  # pragma: no cover - required by Handler API
        return None

    def close(self):
        self.was_closed = True
        super().close()


def _clear_handlers():
    for handler in list(decorators.logger.handlers):
        decorators.logger.removeHandler(handler)
        handler.close()


def test_ensure_logging_configured_removes_closed_handlers():
    _clear_handlers()
    stale = _ClosedHandler()
    decorators.logger.addHandler(stale)

    decorators._ensure_logging_configured()

    assert stale.was_closed is True
    assert stale not in decorators.logger.handlers
    _clear_handlers()


def test_timing_decorator_activate_override_path(monkeypatch):
    decorators._function_stats.clear()
    monkeypatch.setattr(decorators.config, "ACTIVATE_TIMING", False)
    monkeypatch.setattr(decorators.config, "LOG_TIMING", False)

    @decorators.timing_decorator(activate_overide=True)
    def sample():
        return 7

    assert sample() == 7
    assert decorators._function_stats["sample"]["count"] == 1


def test_print_summary_returns_early_when_empty(monkeypatch):
    decorators._function_stats.clear()
    called = {"configured": False}
    monkeypatch.setattr(
        decorators,
        "_ensure_logging_configured",
        lambda: called.__setitem__("configured", True),
    )

    decorators.print_summary()

    assert called["configured"] is False


def test_print_summary_logs_aggregated_stats(monkeypatch):
    decorators._function_stats.clear()
    decorators._function_stats["alpha"]["count"] = 2
    decorators._function_stats["alpha"]["total_time"] = 0.5

    messages = []
    monkeypatch.setattr(decorators, "_ensure_logging_configured", lambda: None)
    monkeypatch.setattr(
        decorators.logger, "info", lambda message: messages.append(str(message))
    )

    decorators.print_summary()

    assert any("Execution Time Summary" in message for message in messages)
    assert any("alpha: 2 calls" in message for message in messages)
