from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem, StreamSegment
from OpenPinch.classes import Stream, Zone
from OpenPinch.lib.enums import ST
from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    pinch_design_decomposition as pdm_decomposition,
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
    profile_from_solver_arrays,
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
from tests.strategies.stream_segments import segmented_streams


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


def _segmented_problem(
    *,
    cold_second_duty: float = 100.0,
    segmented_utility: bool = False,
    segmented_cold_utility: bool = False,
    hot_utility_dt_cont: tuple[float, float] = (0.0, 0.0),
    cold_utility_dt_cont: tuple[float, float] = (0.0, 0.0),
) -> PinchProblem:
    hot_utility = (
        {
            "zone": "Site",
            "name": "HU",
            "type": "Hot",
            "segments": [
                {
                    "t_supply": 250.0,
                    "t_target": 225.0,
                    "heat_flow": 50.0,
                    "htc": 2.0,
                    "price": 20.0,
                    "dt_cont": hot_utility_dt_cont[0],
                },
                {
                    "t_supply": 225.0,
                    "t_target": 200.0,
                    "heat_flow": 100.0,
                    "htc": 1.0,
                    "price": 80.0,
                    "dt_cont": hot_utility_dt_cont[1],
                },
            ],
        }
        if segmented_utility
        else {
            "zone": "Site",
            "name": "HU",
            "type": "Hot",
            "t_supply": 250.0,
            "t_target": 249.99,
            "htc": 2.0,
        }
    )
    cold_utility = (
        {
            "zone": "Site",
            "name": "CU",
            "type": "Cold",
            "segments": [
                {
                    "t_supply": 20.0,
                    "t_target": 45.0,
                    "heat_flow": 50.0,
                    "htc": 2.0,
                    "price": 30.0,
                    "dt_cont": cold_utility_dt_cont[0],
                },
                {
                    "t_supply": 45.0,
                    "t_target": 70.0,
                    "heat_flow": 100.0,
                    "htc": 1.0,
                    "price": 90.0,
                    "dt_cont": cold_utility_dt_cont[1],
                },
            ],
        }
        if segmented_cold_utility
        else {
            "zone": "Site",
            "name": "CU",
            "type": "Cold",
            "t_supply": 20.0,
            "t_target": 20.01,
            "htc": 2.0,
        }
    )
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
                hot_utility,
                cold_utility,
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


def test_segmented_utility_arrays_and_cost_profile_use_local_prices_exactly():
    problem = _segmented_problem(segmented_utility=True)
    arrays = problem_to_solver_arrays(problem, 10.0)
    profile = profile_from_solver_arrays(
        arrays,
        side="hot_utility",
        parent_index=0,
        period_index=0,
    )

    assert arrays.arrays["hot_utility_parent_segmented"].tolist() == [True]
    np.testing.assert_allclose(profile.prices, [20.0, 80.0])
    np.testing.assert_allclose(profile.cumulative_costs, [0.0, 1000.0, 9000.0])
    assert profile.cost_at_heat(25.0) == pytest.approx(500.0)
    assert profile.cost_at_heat(50.0) == pytest.approx(1000.0)
    assert profile.cost_at_heat(75.0) == pytest.approx(3000.0)
    assert profile.cost_at_heat(150.0) == pytest.approx(9000.0)


def test_segmented_utility_dt_cont_tensors_and_boundary_mapping_are_local():
    problem = _segmented_problem(
        segmented_utility=True,
        segmented_cold_utility=True,
        hot_utility_dt_cont=(0.2, 0.8),
        cold_utility_dt_cont=(0.7, 0.3),
    )
    arrays = problem_to_solver_arrays(problem, 10.0)
    hot_profile = profile_from_solver_arrays(
        arrays,
        side="hot_utility",
        parent_index=0,
        period_index=0,
    )
    cold_profile = profile_from_solver_arrays(
        arrays,
        side="cold_utility",
        parent_index=0,
        period_index=0,
    )

    np.testing.assert_allclose(
        arrays.arrays["hot_utility_segment_dt_cont_period"][0, 0],
        [2.0, 8.0],
    )
    np.testing.assert_allclose(
        arrays.arrays["cold_utility_segment_dt_cont_period"][0, 0],
        [7.0, 3.0],
    )
    assert hot_profile.temperature_contribution_at_heat(0.0) == pytest.approx(2.0)
    assert hot_profile.temperature_contribution_at_heat(25.0) == pytest.approx(2.0)
    assert hot_profile.temperature_contribution_at_heat(50.0) == pytest.approx(8.0)
    assert hot_profile.temperature_contribution_at_heat(75.0) == pytest.approx(8.0)
    assert cold_profile.temperature_contribution_at_heat(0.0) == pytest.approx(7.0)
    assert cold_profile.temperature_contribution_at_heat(50.0) == pytest.approx(7.0)
    assert cold_profile.temperature_contribution_at_heat(75.0) == pytest.approx(3.0)


def test_flat_utility_dt_cont_keeps_scalar_contribution():
    arrays = problem_to_solver_arrays(_segmented_problem(), 10.0)
    model = StageWiseModel.__new__(StageWiseModel)
    model.solver_arrays = arrays
    model.T_hu_cont_period = arrays.arrays["T_hu_cont_period"]
    model.T_cu_cont_period = arrays.arrays["T_cu_cont_period"]

    assert model._utility_outlet_temperature_contribution("hot", 0, 0, 25.0) == 5.0
    assert model._utility_outlet_temperature_contribution("cold", 0, 0, 25.0) == 5.0


def test_segmented_utility_dt_cont_mapping_is_built_into_stagewise_constraints():
    arrays = problem_to_solver_arrays(
        _segmented_problem(
            segmented_utility=True,
            segmented_cold_utility=True,
            hot_utility_dt_cont=(0.2, 0.8),
            cold_utility_dt_cont=(0.7, 0.3),
        ),
        10.0,
    )

    model = StageWiseModel(
        name="utility-dt-cont",
        framework="TDM",
        solver="apopt",
        solver_arrays=arrays,
        stages=1,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total utility",
        non_isothermal_model=False,
        integers=False,
        tol=1e-3,
    )

    assert model.T_hu_in_cont_by_period == [2.0]
    assert model.T_cu_in_cont_by_period == [7.0]
    assert len(model.T_hu_out_cont_by_period) == 1
    assert len(model.T_cu_out_cont_by_period) == 1
    assert len(model.T_hu_solved_out_by_period) == 1
    assert len(model.T_cu_solved_out_by_period) == 1
    contribution_mappings = [
        mapping
        for mapping in model._piecewise_active_mappings
        if "segment_index_at_heat" in mapping
    ]
    assert len(contribution_mappings) == 2
    assert contribution_mappings[0]["segment_index_at_heat"](50.0) == 1
    assert contribution_mappings[1]["segment_index_at_heat"](50.0) == 0


@given(
    st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
)
@settings(max_examples=30)
def test_piecewise_utility_cost_conserves_full_segment_cost(
    first_duty,
    second_duty,
    first_price,
    second_price,
):
    profile = PiecewiseThermalProfile(
        identities=("HU.S1", "HU.S2"),
        temperatures_in=np.array([500.0, 450.0]),
        temperatures_out=np.array([450.0, 400.0]),
        duties=np.array([first_duty, second_duty]),
        heat_capacity_flowrates=np.array([first_duty / 50.0, second_duty / 50.0]),
        heat_transfer_coefficients=np.array([1.0, 1.0]),
        prices=np.array([first_price, second_price]),
    )

    expected = first_price * first_duty + second_price * second_duty
    assert profile.cost_at_heat(profile.total_duty) == pytest.approx(expected)


def test_multiperiod_segmented_utility_cost_profiles_keep_stable_identities():
    def period_value(values, unit):
        return {"values": values, "unit": unit}

    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot process",
                    "t_supply": 200.0,
                    "t_target": 100.0,
                    "heat_flow": period_value([150.0, 150.0], "kW"),
                },
                {
                    "zone": "Site",
                    "name": "Cold process",
                    "t_supply": 50.0,
                    "t_target": 150.0,
                    "heat_flow": period_value([225.0, 225.0], "kW"),
                },
            ],
            "utilities": [
                {
                    "name": "HU",
                    "type": "Hot",
                    "segments": [
                        {
                            "t_supply": period_value([250.0, 260.0], "degC"),
                            "t_target": period_value([225.0, 235.0], "degC"),
                            "heat_flow": period_value([50.0, 100.0], "kW"),
                            "price": period_value([20.0, 40.0], "$/MWh"),
                            "dt_cont": period_value([0.2, 0.6], "delta_degC"),
                        },
                        {
                            "t_supply": period_value([225.0, 235.0], "degC"),
                            "t_target": period_value([200.0, 210.0], "degC"),
                            "heat_flow": period_value([100.0, 100.0], "kW"),
                            "price": period_value([80.0, 20.0], "$/MWh"),
                            "dt_cont": period_value([0.8, 0.3], "delta_degC"),
                        },
                    ],
                },
                {
                    "name": "CU",
                    "type": "Cold",
                    "t_supply": 20.0,
                    "t_target": 20.01,
                },
            ],
            "options": {
                "PROBLEM_PERIOD_IDS": ["base", "peak"],
                "PROBLEM_PERIOD_WEIGHTS": [1.0, 3.0],
            },
        },
        project_name="Site",
    )
    arrays = problem_to_solver_arrays(problem, 10.0)
    profiles = [
        profile_from_solver_arrays(
            arrays,
            side="hot_utility",
            parent_index=0,
            period_index=period_index,
        )
        for period_index in range(2)
    ]

    assert profiles[0].identities == profiles[1].identities
    np.testing.assert_allclose(profiles[0].cumulative_duties, [0.0, 50.0, 150.0])
    np.testing.assert_allclose(profiles[1].cumulative_duties, [0.0, 100.0, 200.0])
    assert profiles[0].cost_at_heat(75.0) == pytest.approx(3000.0)
    assert profiles[1].cost_at_heat(150.0) == pytest.approx(5000.0)
    np.testing.assert_allclose(profiles[0].temperature_contributions, [2.0, 8.0])
    np.testing.assert_allclose(profiles[1].temperature_contributions, [6.0, 3.0])
    assert profiles[0].temperature_contribution_at_heat(50.0) == pytest.approx(8.0)
    assert profiles[1].temperature_contribution_at_heat(100.0) == pytest.approx(6.0)


def test_pdm_targeting_applies_hen_dtmin_to_every_expanded_segment(monkeypatch):
    problem = _segmented_problem()
    observed = {}

    def capture_targeting_zone(zone, args=None):
        assert args == {"period_id": "0"}
        numeric = zone.process_streams.segment_numeric_view()
        observed["dt_cont"] = numeric.dt_cont.copy()
        observed["t_min_star"] = numeric.t_min_star.copy()
        observed["t_max_star"] = numeric.t_max_star.copy()
        return SimpleNamespace(
            hot_utility_target=10.0,
            cold_utility_target=10.0,
            heat_recovery_target=140.0,
            hot_pinch=100.0,
            cold_pinch=90.0,
        )

    monkeypatch.setattr(
        pdm_decomposition,
        "compute_direct_integration_targets",
        capture_targeting_zone,
    )

    pdm_decomposition._calculate_openpinch_targets(problem, dTmin=10.0)

    np.testing.assert_allclose(observed["dt_cont"], [5.0, 5.0, 5.0, 5.0])
    np.testing.assert_allclose(
        observed["t_min_star"],
        [145.0, 95.0, 55.0, 105.0],
    )
    np.testing.assert_allclose(
        observed["t_max_star"],
        [195.0, 145.0, 105.0, 155.0],
    )
    for stream in problem.master_zone.process_streams:
        assert float(stream.dt_cont.to("delta_degC").value) == 0.0
        assert all(
            float(segment.dt_cont.to("delta_degC").value) == 0.0
            for segment in stream.segments
        )


def test_pdm_dt_cont_minimum_is_applied_per_segment_and_period():
    stream = Stream(
        name="Multiperiod segmented hot stream",
        segments=[
            StreamSegment(
                t_supply=[200.0, 210.0],
                t_target=[150.0, 160.0],
                heat_flow=[50.0, 60.0],
                dt_cont=[1.0, 9.0],
            ),
            StreamSegment(
                t_supply=[150.0, 160.0],
                t_target=[100.0, 110.0],
                heat_flow=[100.0, 120.0],
                dt_cont=[8.0, 2.0],
            ),
        ],
    )
    zone = SimpleNamespace(all_streams=(stream,), dt_cont_multiplier=2.0)

    pdm_decomposition._apply_hen_dt_cont_convention(zone, dTmin=14.0)

    assert zone.dt_cont_multiplier == 1.0
    np.testing.assert_allclose(
        stream.segments[0].dt_cont.to("delta_degC").period_values,
        [7.0, 9.0],
    )
    np.testing.assert_allclose(
        stream.segments[1].dt_cont.to("delta_degC").period_values,
        [8.0, 7.0],
    )
    np.testing.assert_allclose(
        stream.dt_cont.to("delta_degC").period_values,
        [7.0, 9.0],
    )


@given(
    segmented_streams(),
    st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=30)
def test_pdm_dt_cont_minimum_holds_for_all_generated_segments(stream, dTmin):
    minimum = dTmin / 2.0
    original = []
    for index in range(stream.segment_count):
        value = minimum / 2.0 if index % 2 == 0 else minimum * 2.0 + index
        stream.update_segment(index, dt_cont=value)
        original.append(value)

    zone = Zone()
    destination = zone.hot_streams if stream.type == ST.Hot.value else zone.cold_streams
    destination.add(stream)

    pdm_decomposition._apply_hen_dt_cont_convention(zone, dTmin=dTmin)

    expected = np.maximum(original, minimum)
    numeric = zone.process_streams.segment_numeric_view()
    np.testing.assert_allclose(numeric.dt_cont, expected)
    np.testing.assert_allclose(
        [
            float(segment.dt_cont.to("delta_degC").value)
            for segment in next(iter(zone.process_streams)).segments
        ],
        expected,
    )
    assert float(next(iter(zone.process_streams)).dt_cont.to("delta_degC").value) == (
        pytest.approx(expected[0])
    )


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
@pytest.mark.parametrize(
    ("cold_second_duty", "expected_load", "expected_cost", "expected_segments"),
    [
        (150.0, 50.0, 1000.0, 1),
        (175.0, 75.0, 3000.0, 2),
    ],
)
def test_apopt_segmented_hot_utility_uses_exact_traversed_segment_cost(
    cold_second_duty,
    expected_load,
    expected_cost,
    expected_segments,
):
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name="segmented-priced-hot-utility",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(
            _segmented_problem(
                cold_second_duty=cold_second_duty,
                segmented_utility=True,
            ),
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
    assert model.Q_hu_total_by_period[0] == pytest.approx(expected_load)
    assert model.hu_cost_total == pytest.approx(expected_cost)
    assert model.hu_cost_total != pytest.approx(
        model.hu_cost_period[0][0] * model.Q_hu_total_by_period[0]
    )
    contributions = model.segment_area_hu_contributions_by_period[0][0]
    assert contributions
    assert len({item.hot_segment_identity for item in contributions}) == (
        expected_segments
    )


@pytest.mark.synthesis
def test_apopt_segmented_cold_utility_uses_exact_traversed_segment_cost():
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name="segmented-priced-cold-utility",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(
            _segmented_problem(
                cold_second_duty=50.0,
                segmented_cold_utility=True,
            ),
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
    assert model.Q_cu_total_by_period[0] == pytest.approx(50.0)
    assert model.cu_cost_total == pytest.approx(1500.0)
    assert model.cu_cost_total != pytest.approx(
        model.cu_cost_period[0][0] * model.Q_cu_total_by_period[0]
    )
    contributions = model.segment_area_cu_contributions_by_period[0][0]
    assert contributions
    assert {item.cold_segment_identity for item in contributions} == {
        model.solver_arrays.stream_identities["cold_utility_segments"][0]
    }


@pytest.mark.synthesis
def test_ipopt_segmented_utility_cost_stabilises_or_reports_guidance():
    pytest.importorskip("gekko")
    model = StageWiseModel(
        name="segmented-priced-hot-utility-ipopt",
        framework="TDM",
        solver="ipopt-GEKKO",
        solver_arrays=problem_to_solver_arrays(
            _segmented_problem(
                cold_second_duty=150.0,
                segmented_utility=True,
            ),
            10.0,
        ),
        stages=1,
        dTmin=10.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)

    assert model._piecewise_active_mappings
    if model.mSuccess == 1:
        assert model.Q_hu_total_by_period[0] == pytest.approx(50.0)
        assert model.hu_cost_total == pytest.approx(1000.0)
    else:
        assert "use APOPT or Couenne" in model.solver_run.failure_reason


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
