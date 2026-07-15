from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.extraction import (
    extract_heat_exchanger_network,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.piecewise import (
    PiecewiseThermalProfile,
    duty_aligned_area_contributions,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.pinch_design import (
    PinchDecompModel,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.stagewise import (
    StageWiseModel,
    _check_area_costs,
)


def _profile(prefix: str, *, hot: bool) -> PiecewiseThermalProfile:
    if hot:
        temperatures_in = np.array([473.15, 423.15])
        temperatures_out = np.array([423.15, 373.15])
        htcs = np.array([2.0, 1.0])
    else:
        temperatures_in = np.array([323.15, 373.15])
        temperatures_out = np.array([373.15, 423.15])
        htcs = np.array([1.0, 2.0])
    return PiecewiseThermalProfile(
        identities=(f"{prefix}.S1", f"{prefix}.S2"),
        temperatures_in=temperatures_in,
        temperatures_out=temperatures_out,
        duties=np.array([50.0, 100.0]),
        heat_capacity_flowrates=np.array([1.0, 2.0]),
        heat_transfer_coefficients=htcs,
    )


def _segmented_problem(*, cold_second_duty: float = 100.0) -> PinchProblem:
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot parent",
                    "segments": [
                        {
                            "t_supply": 200.0,
                            "t_target": 150.0,
                            "heat_flow": 50.0,
                            "htc": 2.0,
                        },
                        {
                            "t_supply": 150.0,
                            "t_target": 100.0,
                            "heat_flow": 100.0,
                            "htc": 1.0,
                        },
                    ],
                },
                {
                    "zone": "Site",
                    "name": "Cold parent",
                    "segments": [
                        {
                            "t_supply": 50.0,
                            "t_target": 100.0,
                            "heat_flow": 50.0,
                            "htc": 1.0,
                        },
                        {
                            "t_supply": 100.0,
                            "t_target": 150.0,
                            "heat_flow": cold_second_duty,
                            "htc": 2.0,
                        },
                    ],
                },
            ],
            "utilities": [
                {
                    "zone": "Site",
                    "name": "HU",
                    "type": "Hot",
                    "t_supply": 250.0,
                    "t_target": 249.99,
                    "htc": 2.0,
                },
                {
                    "zone": "Site",
                    "name": "CU",
                    "type": "Cold",
                    "t_supply": 20.0,
                    "t_target": 20.01,
                    "htc": 2.0,
                },
            ],
        },
        project_name="Site",
    )


def test_solver_arrays_keep_parent_axes_and_add_ordered_segment_tensors():
    arrays = problem_to_solver_arrays(_segmented_problem(), 10.0)

    assert arrays.arrays["hot_segment_count"].tolist() == [2]
    assert arrays.arrays["cold_segment_count"].tolist() == [2]
    assert arrays.arrays["hot_segment_mask_period"].shape == (1, 1, 2)
    np.testing.assert_allclose(
        arrays.arrays["hot_segment_cumulative_duty_period"][0, 0],
        [0.0, 50.0, 150.0],
    )
    hot_parent_key = arrays.stream_identities["hot_process_streams"][0]
    assert arrays.arrays["hot_segment_identities"][0].tolist() == [
        f"{hot_parent_key}.S1",
        f"{hot_parent_key}.S2",
    ]
    assert arrays.arrays["f_h_period"][0, 0] == 0.0
    assert arrays.arrays["f_c_period"][0, 0] == 0.0
    assert len(arrays.axis_maps["hot_process_streams"]) == 1
    assert len(arrays.axis_maps["cold_process_streams"]) == 1


def test_duty_aligned_area_slices_use_local_htcs_and_sum_exactly():
    contributions = duty_aligned_area_contributions(
        _profile("Hot", hot=True),
        _profile("Cold", hot=False),
        duty=120.0,
        hot_inlet_temperature=473.15,
        cold_inlet_temperature=323.15,
        period="normal",
    )

    assert [item.duty for item in contributions] == pytest.approx([50.0, 20.0, 50.0])
    assert [item.hot_segment_identity for item in contributions] == [
        "Hot.S1",
        "Hot.S2",
        "Hot.S2",
    ]
    assert sum(item.duty for item in contributions) == pytest.approx(120.0)
    assert [item.lmtd for item in contributions] == pytest.approx(
        [51.4924769228, 40.0, 51.4924769228]
    )
    assert sum(item.area for item in contributions) == pytest.approx(3.6630468947)


def test_piecewise_profile_rejects_discontinuity_and_out_of_range_lookup():
    with pytest.raises(ValueError, match="continuous"):
        PiecewiseThermalProfile(
            identities=("S1", "S2"),
            temperatures_in=np.array([473.15, 420.0]),
            temperatures_out=np.array([423.15, 373.15]),
            duties=np.array([50.0, 100.0]),
            heat_capacity_flowrates=np.array([1.0, 2.0]),
            heat_transfer_coefficients=np.array([2.0, 1.0]),
        )

    with pytest.raises(ValueError, match="outside"):
        _profile("Hot", hot=True).heat_at_temperature(500.0)


def test_design_area_uses_maximum_period_total_not_segment_maxima():
    normal = duty_aligned_area_contributions(
        _profile("Hot", hot=True),
        _profile("Cold", hot=False),
        duty=120.0,
        hot_inlet_temperature=473.15,
        cold_inlet_temperature=323.15,
        period="normal",
    )
    peak = tuple(
        item.model_copy(
            update={
                "period": "peak",
                "lmtd": item.lmtd / 2.0,
                "area": item.area * 2.0,
            }
        )
        for item in normal
    )

    normal_area = sum(item.area for item in normal)
    peak_area = sum(item.area for item in peak)
    assert max(normal_area, peak_area) == pytest.approx(2.0 * normal_area)


@given(st.floats(min_value=1.0, max_value=149.0, allow_nan=False))
@settings(max_examples=30)
def test_slice_duty_and_area_sum_invariants(duty):
    contributions = duty_aligned_area_contributions(
        _profile("Hot", hot=True),
        _profile("Cold", hot=False),
        duty=duty,
        hot_inlet_temperature=473.15,
        cold_inlet_temperature=323.15,
        period="normal",
    )

    assert sum(item.duty for item in contributions) == pytest.approx(duty)
    assert sum(item.area for item in contributions) == pytest.approx(
        sum(item.duty / item.overall_htc / item.lmtd for item in contributions)
    )


def test_extraction_retains_one_parent_exchanger_with_segment_area_details():
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

    network = extract_heat_exchanger_network(solved, arrays, run_id="segmented")

    assert len(network.exchangers) == 1
    exchanger = network.exchangers[0]
    assert exchanger.source_stream.endswith("Hot parent")
    assert exchanger.sink_stream.endswith("Cold parent")
    assert exchanger.duty == pytest.approx(120.0)
    assert exchanger.area == pytest.approx(3.6630468947)
    assert len(exchanger.segment_area_contributions) == 3


def test_verification_keeps_segment_failures_without_aggregate_cp_metadata():
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
    network = extract_heat_exchanger_network(solved, arrays, run_id="segmented")
    exchanger = network.exchangers[0]
    first, *rest = exchanger.segment_area_contributions
    bad_exchanger = exchanger.model_copy(
        update={
            "segment_area_contributions": (
                first.model_copy(update={"duty": first.duty + 5.0}),
                *rest,
            )
        }
    )
    network = network.model_copy(
        update={"exchangers": (bad_exchanger,), "source_metadata": {}}
    )

    assert any(
        "segment duty" in failure for failure in verify_network_feasibility(network)
    )


@pytest.mark.synthesis
@pytest.mark.parametrize("non_isothermal_model", [False, True])
def test_apopt_stagewise_uses_parent_heat_coordinates_and_segment_area_totals(
    non_isothermal_model,
):
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name=f"segmented-parent-noniso-{non_isothermal_model}",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(_segmented_problem(), 10.0),
        stages=1,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=non_isothermal_model,
        integers=True,
        tol=1e-6,
    )

    model.optimise(False)

    assert model.mSuccess == 1
    assert model.Q_r_total == pytest.approx(150.0)
    assert model.Q_coordinate_h_by_period
    contributions = model.segment_area_contributions_by_period[0][0][0][0]
    assert sum(item.duty for item in contributions) == pytest.approx(150.0)
    assert model.area_r[0][0][0] == pytest.approx(
        sum(item.area for item in contributions)
    )
    assert model.dqda[0][0][0] > 0.0
    assert _check_area_costs(model)


@pytest.mark.synthesis
def test_ipopt_stagewise_stabilises_or_reports_integer_solver_guidance():
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name="segmented-parent-ipopt",
        framework="TDM",
        solver="ipopt-GEKKO",
        solver_arrays=problem_to_solver_arrays(_segmented_problem(), 10.0),
        stages=1,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)

    assert model._piecewise_active_mappings
    if model.mSuccess == 1:
        assert model.Q_r_total == pytest.approx(150.0)
    else:
        assert "use APOPT or Couenne" in model.solver_run.failure_reason


@pytest.mark.synthesis
def test_apopt_utility_exchanger_uses_segment_area_contributions():
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name="segmented-parent-utility",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(
            _segmented_problem(cold_second_duty=50.0),
            10.0,
        ),
        stages=1,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=True,
        tol=1e-6,
    )

    model.optimise(False)

    assert model.mSuccess == 1
    contributions = model.segment_area_cu_contributions_by_period[0][0]
    assert contributions
    assert sum(item.duty for item in contributions) == pytest.approx(
        model.Q_cu_total_by_period[0]
    )
    assert model.area_cu[0] == pytest.approx(sum(item.area for item in contributions))
    assert model.verify() == (True, [])


@pytest.mark.synthesis
def test_apopt_pdm_clips_segment_profile_without_splitting_parent_identity():
    pytest.importorskip("gekko")
    problem = _segmented_problem()
    arrays = problem_to_solver_arrays(problem, 10.0)
    decomposition = build_pinch_design_decomposition(
        problem,
        10.0,
        pinch_location="above",
    )
    model = PinchDecompModel(
        name="segmented-pdm-above",
        framework="PDM",
        solver="apopt",
        solver_arrays=arrays,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-6,
        pinch_loc="above",
        pinch_decomposition=decomposition,
        stage_selection="automated",
    )

    model.optimise(False)

    assert model.mSuccess == 1
    assert model.I == model.J == 1
    assert model.Q_r_total == pytest.approx(150.0)
    assert model.segment_area_contributions_by_period
