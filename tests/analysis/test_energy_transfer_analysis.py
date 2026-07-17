"""Tests for energy-transfer diagram and surplus/deficit table helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.energy_transfer.cascade import (
    compile_temperature_intervals as _compile_temperature_intervals,
)
from OpenPinch.analysis.energy_transfer.cascade import (
    cross_pinch_flags as _cross_pinch_flags,
)
from OpenPinch.analysis.energy_transfer.cascade import (
    has_problem_table_values as _has_problem_table_values,
)
from OpenPinch.analysis.energy_transfer.cascade import (
    pinch_weights as _pinch_weights,
)
from OpenPinch.analysis.energy_transfer.cascade import (
    stack_cascades as _stack_cascades,
)
from OpenPinch.analysis.energy_transfer.cascade import (
    transpose_operation_cascades as _transpose_operation_cascades,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    create_energy_transfer_diagram,
    create_heat_surplus_deficit_table,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    empty_diagram as _empty_diagram,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    get_base_problem_table as _get_base_problem_table,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    operation_mode as _operation_mode,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    operation_name as _operation_name,
)
from OpenPinch.analysis.energy_transfer.diagram import (
    pinch_temperatures as _pinch_temperatures,
)
from OpenPinch.analysis.energy_transfer.selection import (
    ensure_base_target as _ensure_energy_transfer_base_target,
)
from OpenPinch.analysis.energy_transfer.selection import (
    normalize_base_target_type as _normalize_energy_transfer_base_target_type,
)
from OpenPinch.analysis.energy_transfer.service import (
    run_energy_transfer_analysis_service,
)
from OpenPinch.domain.enums import GT, PT, TT
from OpenPinch.domain.problem_table import ProblemTable


def _sample_problem_table() -> ProblemTable:
    return ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0],
            PT.CP_NET: [0.0, 1.0, -1.2],
            PT.H_NET: [15.0, 65.0, 5.0],
        }
    )


def test_create_heat_surplus_deficit_table_reports_operation_delta_hnet():
    table = create_heat_surplus_deficit_table(_sample_problem_table())

    assert table == [
        {
            "temperature": 200.0,
            "interval": 0.0,
            "Process": 0.0,
            "total_delta_hnet": 0.0,
            "heat_surplus": 0.0,
            "heat_deficit": 0.0,
        },
        {
            "temperature": 150.0,
            "interval": 1.0,
            "Process": 50.0,
            "total_delta_hnet": 50.0,
            "heat_surplus": 0.0,
            "heat_deficit": 50.0,
        },
        {
            "temperature": 100.0,
            "interval": 2.0,
            "Process": -60.0,
            "total_delta_hnet": -60.0,
            "heat_surplus": 60.0,
            "heat_deficit": 0.0,
        },
    ]


def test_create_energy_transfer_diagram_builds_stacked_operation_profile():
    pt = _sample_problem_table()
    diagram = create_energy_transfer_diagram(pt)

    assert diagram["temperatures"] == [200.0, 150.0, 100.0]
    assert len(diagram["operations"]) == 1
    operation = diagram["operations"][0]
    assert operation["interval_heat"] == [50.0, -60.0]
    assert operation["cascade_heat"] == [15.0, 65.0, 5.0]
    assert operation["stacked_heat"] == [15.0, 65.0, 5.0]


def test_create_energy_transfer_diagram_prefers_saved_gcc_payload():
    fallback_pt = ProblemTable(
        {
            PT.T: [200.0, 100.0],
            PT.CP_NET: [0.0, 0.0],
            PT.H_NET: [999.0, 999.0],
        }
    )
    gcc = ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0],
            PT.H_NET: [0.0, 25.0, -10.0],
        }
    )
    target = SimpleNamespace(
        zone_name="NestedUnit",
        type=TT.DI.value,
        pt=fallback_pt,
        graphs={GT.GCC.value: gcc},
        hot_utility_target=0.0,
        cold_utility_target=0.0,
    )

    diagram = create_energy_transfer_diagram([target])

    assert diagram["temperatures"] == [200.0, 150.0, 100.0]
    assert diagram["operations"][0]["name"] == "NestedUnit"
    assert diagram["operations"][0]["interval_heat"] == [25.0, -35.0]
    assert diagram["operations"][0]["cascade_heat"] == [0.0, 25.0, -10.0]


def test_empty_energy_transfer_inputs_return_empty_diagram_and_table():
    base_target = SimpleNamespace(
        name="Base",
        type=TT.DI.value,
        hot_pinch=120.0,
        cold_pinch=90.0,
    )

    diagram = create_energy_transfer_diagram([], base_target=base_target)

    assert diagram == _empty_diagram(base_target)
    assert (
        create_heat_surplus_deficit_table({"temperatures": [], "operations": []}) == []
    )


def test_energy_transfer_service_handles_period_context_and_error_paths():
    class FakeZone:
        def __init__(self):
            self.name = "Plant"
            self.targets = {}
            self.subzones = {}
            self.period_ids = {"peak": 1}

        def add_target(self, target):
            self.targets[target.type] = target

    zone = FakeZone()
    target = SimpleNamespace(
        type=TT.DI.value,
        period_id="peak",
        period_idx=1,
    )
    zone.targets[TT.DI.value] = target

    result_target = SimpleNamespace(type=TT.ET.value)
    out = run_energy_transfer_analysis_service(
        zone,
        {"period_id": "peak"},
        refresh_services={},
        compute_func=lambda base_target, source_targets=None: result_target,
    )

    assert out is zone
    assert zone._selected_period_id == "peak"
    assert zone.targets[TT.ET.value] is result_target

    with pytest.raises(ValueError, match="Unsupported energy-transfer"):
        _normalize_energy_transfer_base_target_type("unsupported")

    with pytest.raises(RuntimeError, match="could not produce base target"):
        run_energy_transfer_analysis_service(
            FakeZone(),
            {"base_target_type": TT.TS.value},
            refresh_services={TT.TS.value: lambda target_zone, args=None: target_zone},
            compute_func=lambda base_target, source_targets=None: result_target,
        )

    with pytest.raises(RuntimeError, match="could not find a compatible target"):
        run_energy_transfer_analysis_service(
            FakeZone(),
            None,
            refresh_services={},
            compute_func=lambda base_target, source_targets=None: result_target,
        )


def test_ensure_energy_transfer_base_target_reports_missing_and_stale_refreshes():
    zone = SimpleNamespace(
        name="Plant",
        targets={},
        subzones={},
        period_ids={"peak": 0},
    )

    assert (
        _ensure_energy_transfer_base_target(
            zone,
            target_type=TT.DI.value,
            refresh_args={},
            compare_args={},
            refresh_services={},
        )
        is None
    )
    assert (
        _ensure_energy_transfer_base_target(
            zone,
            target_type=TT.TS.value,
            refresh_args={},
            compare_args={},
            refresh_services={TT.TS.value: lambda target_zone, args=None: None},
        )
        is None
    )

    def stale_refresh(target_zone, args=None):
        target_zone.targets[TT.DI.value] = SimpleNamespace(
            type=TT.DI.value,
            period_id="off-peak",
        )

    assert (
        _ensure_energy_transfer_base_target(
            zone,
            target_type=TT.DI.value,
            refresh_args={"period_id": "peak"},
            compare_args={"period_id": "peak"},
            refresh_services={TT.DI.value: stale_refresh},
        )
        is None
    )


def test_energy_transfer_interval_helpers_handle_degenerate_and_one_point_tables():
    empty_pt = ProblemTable({PT.T: [], PT.H_NET: []})
    temperatures = _compile_temperature_intervals(
        [{"name": "Empty", "mode": "R", "pt": empty_pt}],
        base_table=None,
    )
    assert temperatures.size == 0

    one_point = ProblemTable({PT.T: [200.0], PT.H_NET: [25.0]})
    names, modes, interval_heat, cascades = _transpose_operation_cascades(
        [{"name": "One", "mode": "R", "pt": one_point}],
        np.array([200.0, 150.0]),
    )

    assert names == ["One"]
    assert modes == ["R"]
    np.testing.assert_allclose(interval_heat, np.array([[0.0]]))
    np.testing.assert_allclose(cascades, np.array([[25.0, 25.0]]))


def test_energy_transfer_pinch_sorting_and_stack_helpers_cover_edge_branches():
    weights = _pinch_weights(
        np.array([200.0, 120.0, 60.0]),
        hot_pinch=150.0,
        cold_pinch=90.0,
    )
    assert weights[0] < 1.0
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] < 1.0

    crosses = _cross_pinch_flags(
        np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([180.0, 120.0, 80.0]),
        hot_pinch=130.0,
        cold_pinch=90.0,
    )
    np.testing.assert_array_equal(crosses, np.array([True, False]))
    np.testing.assert_array_equal(_stack_cascades(np.array([])), np.array([]))


def test_energy_transfer_target_naming_modes_and_problem_table_guards():
    assert _pinch_temperatures(
        SimpleNamespace(hot_pinch=140.0, cold_pinch=None),
        base_table=None,
    ) == (140.0, None)
    assert _operation_name(
        SimpleNamespace(zone_name=None, name=None, type=TT.DI.value), 2
    ) == ("Operation 3")
    assert (
        _operation_name(
            SimpleNamespace(zone_name="Plant/Direct Integration", type=TT.DI.value),
            0,
        )
        == "Plant"
    )
    assert _operation_mode(SimpleNamespace(hot_utility_target=1.0)) == "H"
    assert (
        _operation_mode(
            SimpleNamespace(hot_utility_target=0.0, cold_utility_target=2.0)
        )
        == "C"
    )

    with pytest.raises(TypeError, match="requires a base target"):
        _get_base_problem_table(SimpleNamespace(pt=None))

    class MissingColumnTable:
        def __getitem__(self, column):
            raise KeyError(column)

    assert _has_problem_table_values(MissingColumnTable(), PT.H_NET) is False
    invalid_pt = ProblemTable({PT.T: [100.0, 80.0]})
    assert _has_problem_table_values(invalid_pt, PT.H_NET) is False
    nan_pt = ProblemTable({PT.H_NET: [np.nan, np.nan]})
    assert _has_problem_table_values(nan_pt, PT.H_NET) is False
