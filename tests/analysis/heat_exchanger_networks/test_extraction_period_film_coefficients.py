"""A stage-wise network records each period's film coefficients.

Verification sizes each period with ``*_heat_transfer_coefficients_by_period``
when a network records them, else with the scalar coefficients, which the
stage-wise extraction took from the first period alone. On the crude preheat
train on ``design_basis="all"`` with film coefficients rising over three
periods (0.5, 1, 2) every evolved candidate was sized at 0.5 in each period,
judged short of area by half and rejected (review finding 26).
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
    extract_heat_exchanger_network,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_segmented_streams import (
    _segmented_problem,
)


def test_a_stagewise_network_records_each_periods_film_coefficients():
    arrays = problem_to_solver_arrays(_segmented_problem(), 10.0)
    solved = SimpleNamespace(
        S=1,
        Q_r=[[[120.0]]],
        Q_r_by_period=[[[[120.0]]]],
        z=[[[1.0]]],
        z_allowed=[[[1]]],
        Q_h=[0.0],
        Q_c=[0.0],
        z_hu=[0.0],
        z_cu=[0.0],
        T_h=[[473.15, 388.15]],
        T_c=[[408.15, 323.15]],
        T_h_by_period=[[[473.15, 388.15]]],
        T_c_by_period=[[[408.15, 323.15]]],
        area_r=[[[999.0]]],
    )

    metadata = extract_heat_exchanger_network(
        solved, arrays, run_id="periods"
    ).source_metadata

    for key, name in (
        ("hot_stream", "htc_h_period"),
        ("cold_stream", "htc_c_period"),
        ("hot_utility", "htc_hu_period"),
        ("cold_utility", "htc_cu_period"),
    ):
        recorded = metadata[f"{key}_heat_transfer_coefficients_by_period"]
        assert recorded == arrays.arrays[name].tolist()
