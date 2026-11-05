"""Settings shared by every environment the suite runs in."""

import os

import pytest
from hypothesis import settings

# Hypothesis loads its own "ci" profile (derandomize=True) when CI is set, so
# --hypothesis-seed would be ignored on GitHub but honoured locally and in
# Pyodide. This profile makes the seed choose the examples everywhere.
# deadline=None: timing depends on the machine, and WebAssembly runs several
# times slower than native CPython. database=None: examples saved by earlier
# local runs would otherwise change what a seeded run tries.
settings.register_profile(
    "openpinch", derandomize=False, deadline=None, database=None, print_blob=True
)
settings.load_profile("openpinch")


# Each pool re-imports OpenPinch in every worker, which costs seconds per solve
# and fights pytest-xdist for cores. The release lanes set this to keep them.
_REAL_POOLS = os.environ.get("OPENPINCH_TEST_REAL_POOLS") == "1"


@pytest.fixture(scope="session", autouse=True)
def _real_pool_worker_count():
    """Run optimisation tasks serially; session scope covers shared fixtures."""
    from OpenPinch.optimisation import execution

    real = execution._pool_worker_count
    if _REAL_POOLS:
        yield real
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "_pool_worker_count", lambda requested, count: 1)
        yield real


@pytest.fixture(autouse=True)
def _pools_where_marked(request, _real_pool_worker_count):
    """Give tests marked ``process_pools`` the real worker count."""
    if request.node.get_closest_marker("process_pools") is None:
        yield
        return
    from OpenPinch.optimisation import execution

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "_pool_worker_count", _real_pool_worker_count)
        yield
