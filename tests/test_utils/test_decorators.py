import logging
import re
import time

from OpenPinch.lib import *
from OpenPinch.utils import *


def get_dummy_function():
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
