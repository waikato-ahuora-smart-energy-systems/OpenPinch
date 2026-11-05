"""The suite runs optimisation tasks serially unless a test is about pools."""

from __future__ import annotations

import os

import pytest

from OpenPinch.optimisation import execution


@pytest.mark.skipif(
    os.environ.get("OPENPINCH_TEST_REAL_POOLS") == "1",
    reason="this lane runs every test with real pools",
)
def test_unmarked_tests_run_optimisation_tasks_serially():
    assert execution._pool_worker_count(4, 4) == 1


@pytest.mark.process_pools
def test_marked_tests_use_the_real_worker_count(monkeypatch):
    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.setattr(execution.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(execution.multiprocessing, "get_start_method", lambda: "fork")

    assert execution._pool_worker_count(4, 4) == 4
