from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.analysis.direct_integration_entry import (
    _add_net_segment_stateful,
    _initialise_utility_index,
)


def test_initialise_utility_index_returns_first_available():
    u1 = Stream("U1", 200, 250, heat_flow=0.0)
    u2 = Stream("U2", 200, 250, heat_flow=75.0)
    idx = _initialise_utility_index([u1, u2], [u1.heat_flow, u2.heat_flow])
    assert idx == 1


def test_add_net_segment_stateful_consumes_residuals():
    u1 = Stream("U1", 200, 250, heat_flow=120.0)
    u2 = Stream("U2", 200, 250, heat_flow=80.0)
    residuals = [u1.heat_flow, u2.heat_flow]
    net_streams = StreamCollection()
    next_idx, next_k = _add_net_segment_stateful(
        400,
        300,
        0,
        150,
        [u1, u2],
        residuals,
        net_streams,
        k=1,
    )
    assert next_idx == 1
    assert next_k == 2
    assert abs(residuals[0]) < tol
    assert abs(residuals[1] - 50) < tol
    assert len(net_streams) == 2
    assert net_streams[0].name == "Segment 1"
    assert net_streams[1].name == "Segment 1-1"
