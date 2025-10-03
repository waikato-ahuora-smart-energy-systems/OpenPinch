import pytest
import pandas as pd
from OpenPinch.lib import * 
from OpenPinch.classes import *
from OpenPinch.analysis.support_methods import *


from OpenPinch.scales.process_analysis import _initialise_utility_selected

def test_initialise_utility_selected_returns_first_positive():
    utils = StreamCollection()
    utils.add(Stream("U1", 200, 250, heat_flow=1.0))
    utils.add(Stream("U2", 200, 250, heat_flow=1.0))
    selected = _initialise_utility_selected(utils)
    assert selected.name == "U1"

def test_initialise_utility_selected_falls_back_to_last():
    utils = StreamCollection()
    utils.add(Stream("U1", 200, 250, heat_flow=0.0))
    utils.add(Stream("U2", 200, 250, heat_flow=0.0))
    selected = _initialise_utility_selected(utils)
    assert selected.name == "U2"


from OpenPinch.scales.process_analysis import _advance_utility_if_needed

def test_advance_utility_reduces_current_heat_flow():
    u1 = Stream("U1", 200, 250, heat_flow=150.0)
    utils = StreamCollection()
    utils.add(u1)
    result = _advance_utility_if_needed(50, u1, utils)
    assert result.name == "U1"
    assert abs(result.heat_flow - 100.0) < tol

def test_advance_utility_moves_to_next():
    u1 = Stream("U1", 200, 250, heat_flow=100.0)
    u2 = Stream("U2", 200, 250, heat_flow=200.0)
    utils = StreamCollection()
    utils.add_many([u1, u2])
    result = _advance_utility_if_needed(150, u1, utils)
    assert result.name == "U2"
    assert abs(result.heat_flow - 200.0) < tol

def test_advance_utility_falls_back_to_last():
    u1 = Stream("U1", 200, 250, heat_flow=50.0)
    u2 = Stream("U2", 200, 250, heat_flow=0.0)
    utils = StreamCollection()
    utils.add_many([u1, u2])
    result = _advance_utility_if_needed(100, u1, utils)
    assert result.name == "U2"  # fallback


from OpenPinch.scales.process_analysis import _add_net_segment

def test_add_net_segment_single_segment():
    util = Stream("CU1", 200, 250, heat_flow=500)
    utils = StreamCollection()
    utils.add(util)
    net_streams = StreamCollection()

    _add_net_segment(400, 300, util, 400, utils, net_streams, k=1)

    assert len(net_streams) == 1
    seg = net_streams[0]
    assert abs(seg.heat_flow - 400) < tol
    assert seg.name == "Segment 1"

def test_add_net_segment_recursive_split():
    u1 = Stream("CU1", 200, 250, heat_flow=300)
    u2 = Stream("CU2", 200, 250, heat_flow=300)
    utils = StreamCollection()
    utils.add_many([u1, u2])
    net_streams = StreamCollection()

    _add_net_segment(400, 300, u1, 500, utils, net_streams, k=1)

    assert len(net_streams) == 2
    assert net_streams[0].name == "Segment 1"
    assert net_streams[1].name == "Segment 1-1"
    assert abs(net_streams[0].heat_flow - 300) < tol
    assert abs(net_streams[1].heat_flow - 200) < tol

def test_add_net_segment_temperature_interpolation():
    util = Stream("CU1", 200, 250, heat_flow=100)
    utils = StreamCollection()
    utils.add(util)
    net_streams = StreamCollection()

    _add_net_segment(400, 300, util, 50, utils, net_streams, k=1)

    seg = net_streams[0]
    assert seg.t_supply == 400
    assert seg.t_target == 300  # Interpolated

def test_add_net_segment_hot_utility_direction():
    util = Stream("HU1", 400, 300, heat_flow=100)
    utils = StreamCollection()
    utils.add(util)
    net_streams = StreamCollection()

    _add_net_segment(350, 290, util, -50, utils, net_streams, k=1)

    seg = net_streams[0]
    assert seg.t_target == 350
    assert seg.t_supply == 290
    assert abs(seg.heat_flow - 50) < tol
