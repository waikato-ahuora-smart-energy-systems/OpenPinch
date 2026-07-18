from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.common._shared.plotting as hp_plotting
import OpenPinch.analysis.heat_pumps.common._shared.streams as hp_streams
import OpenPinch.analysis.heat_pumps.service as hp
from OpenPinch.analysis.graphs.service import _create_graph_set
from OpenPinch.analysis.heat_pumps.common.load_selection import (
    resolve_hpr_target_load,
)
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRThermoArtifacts,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import (
    GraphType,
    HeatPumpAndRefrigerationCycle,
    ProblemTableLabel,
    TargetType,
    ZoneType,
)
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.zone import Zone

from .helpers import _base_args, _patch_output_model_validate, _pt_with_hnet


def _make_base_utility_collections(
    *,
    hot_flow: float = 40.0,
    cold_flow: float = 30.0,
) -> tuple[StreamCollection, StreamCollection]:
    hot_utilities = StreamCollection()
    hot_utilities.add(
        Stream(
            name="Steam",
            supply_temperature=120.0,
            target_temperature=80.0,
            heat_flow=hot_flow,
            delta_t_contribution=0.0,
        )
    )
    cold_utilities = StreamCollection()
    cold_utilities.add(
        Stream(
            name="Cooling Water",
            supply_temperature=20.0,
            target_temperature=60.0,
            heat_flow=cold_flow,
            delta_t_contribution=0.0,
        )
    )
    return hot_utilities, cold_utilities


@pytest.mark.parametrize(
    ("q_amb_hot", "q_amb_cold", "expected_hot", "expected_cold", "expected_w_air"),
    [
        (
            20.0,
            0.0,
            np.array([-2.0, 3.0]),
            np.array([3.0, 3.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            20.0,
            np.array([2.0, 2.0]),
            np.array([7.0, 2.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            0.0,
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([1.0, 1.0]),
        ),
    ],
)
def test_calc_heat_pump_and_refrigeration_cascade_branches(
    monkeypatch, q_amb_hot, q_amb_cold, expected_hot, expected_cold, expected_w_air
):
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 60.0],
            ProblemTableLabel.H_NET_A: [1.0, 1.0],
            ProblemTableLabel.H_NET_HOT: [2.0, 2.0],
            ProblemTableLabel.H_NET_COLD: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda **_kwargs: ProblemTable({ProblemTableLabel.T: [120.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **_kwargs: {
            "T_col": np.array([120.0, 60.0]),
            "updates": {
                ProblemTableLabel.H_NET_UT: np.array([0.0, 0.0]),
                ProblemTableLabel.H_HOT_UT: np.array([0.0, 0.0]),
                ProblemTableLabel.H_COLD_UT: np.array([0.0, 0.0]),
            },
        },
    )
    ambient_hot = np.array([-4.0, 1.0]) if q_amb_hot > 0.0 else np.zeros(2)
    ambient_cold = np.array([4.0, -1.0]) if q_amb_cold > 0.0 else np.zeros(2)
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: _pt_with_hnet(
            4.0,
            -1.0,
            h_hot=ambient_hot,
            h_cold=ambient_cold,
        ),
    )

    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        Q_amb_hot=q_amb_hot,
        Q_amb_cold=q_amb_cold,
        amb_streams=hp_streams.get_ambient_air_stream(
            q_amb_hot, q_amb_cold, _base_args()
        ),
    )
    out = hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
        period_idx=0,
    )
    assert isinstance(out, ProblemTable)
    np.testing.assert_allclose(out[ProblemTableLabel.H_NET_HOT], expected_hot)
    np.testing.assert_allclose(out[ProblemTableLabel.H_NET_COLD], expected_cold)
    np.testing.assert_allclose(out[ProblemTableLabel.H_NET_W_AIR], expected_w_air)


def test_calc_hpr_cascade_uses_shared_temperature_intervals_for_hpr_and_air(
    monkeypatch,
):
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 60.0],
            ProblemTableLabel.H_NET_A: [1.0, 1.0],
            ProblemTableLabel.H_NET_HOT: [2.0, 2.0],
            ProblemTableLabel.H_NET_COLD: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda **_kwargs: ProblemTable({ProblemTableLabel.T: [120.0, 100.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 90.0, 60.0],
                ProblemTableLabel.H_NET: [6.0, 3.0, 0.0],
                ProblemTableLabel.H_NET_HOT: [0.0, 3.0, 6.0],
                ProblemTableLabel.H_NET_COLD: [6.0, 3.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **kwargs: {
            "T_col": np.array(kwargs["T_int_vals"], dtype=float),
            "updates": {
                ProblemTableLabel.H_NET_UT: np.zeros(len(kwargs["T_int_vals"])),
                ProblemTableLabel.H_HOT_UT: np.zeros(len(kwargs["T_int_vals"])),
                ProblemTableLabel.H_COLD_UT: np.zeros(len(kwargs["T_int_vals"])),
            },
        },
    )

    amb_streams = StreamCollection()
    amb_streams.add(
        Stream(
            name="Air", supply_temperature=50.0, target_temperature=80.0, heat_flow=10.0
        )
    )
    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        Q_amb_hot=10.0,
        Q_amb_cold=0.0,
        amb_streams=amb_streams,
    )

    out = hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
    )

    np.testing.assert_allclose(
        out[ProblemTableLabel.T], np.array([120.0, 100.0, 90.0, 60.0])
    )
    np.testing.assert_allclose(
        out[ProblemTableLabel.H_NET_W_AIR], np.array([7.0, 5.0, 4.0, 1.0])
    )
    np.testing.assert_allclose(
        out[ProblemTableLabel.H_NET_HOT], np.array([2.0, 4.0, 5.0, 8.0])
    )
    np.testing.assert_allclose(
        out[ProblemTableLabel.H_NET_COLD], np.array([9.0, 7.0, 6.0, 3.0])
    )


@pytest.mark.parametrize("period_idx", [None, 3])
def test_calc_hpr_cascade_forwards_period_idx_to_nested_helpers(
    monkeypatch, period_idx
):
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 60.0],
            ProblemTableLabel.H_NET_A: [1.0, 1.0],
            ProblemTableLabel.H_NET_HOT: [2.0, 2.0],
            ProblemTableLabel.H_NET_COLD: [3.0, 3.0],
        }
    )
    calls = {}

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda **kwargs: (
            calls.__setitem__("grid_idx", kwargs["period_idx"])
            or ProblemTable({ProblemTableLabel.T: [120.0, 60.0]})
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("air_idx", kwargs["period_idx"])
            or ProblemTable(
                {
                    ProblemTableLabel.T: [120.0, 60.0],
                    ProblemTableLabel.H_NET: [0.0, 0.0],
                    ProblemTableLabel.H_NET_HOT: [0.0, 0.0],
                    ProblemTableLabel.H_NET_COLD: [0.0, 0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("utility_idx", kwargs["period_idx"])
            or {
                "T_col": np.array(kwargs["T_int_vals"], dtype=float),
                "updates": {
                    ProblemTableLabel.H_NET_UT: np.zeros(len(kwargs["T_int_vals"])),
                    ProblemTableLabel.H_HOT_UT: np.zeros(len(kwargs["T_int_vals"])),
                    ProblemTableLabel.H_COLD_UT: np.zeros(len(kwargs["T_int_vals"])),
                },
            }
        ),
    )

    amb_streams = StreamCollection()
    amb_streams.add(
        Stream(
            name="Air", supply_temperature=50.0, target_temperature=80.0, heat_flow=10.0
        )
    )
    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        amb_streams=amb_streams,
    )

    hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
        period_idx=period_idx,
    )

    assert calls == {
        "grid_idx": period_idx,
        "air_idx": period_idx,
        "utility_idx": period_idx,
    }


def test_get_hpr_targets_forwards_selected_idx_to_preprocessing(monkeypatch):
    captured = {}
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda *args, period_idx=0, **kwargs: (
            captured.__setitem__("period_idx", period_idx)
            or _base_args(period_idx=period_idx)
        ),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
        lambda args: SimpleNamespace(
            to_output_fields=lambda: {"period_idx": args.period_idx}
        ),
    )

    out = hp._get_hpr_targets(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        config=SimpleNamespace(
            HPR_TYPE=HeatPumpAndRefrigerationCycle.CascadeCarnot.value
        ),
        is_heat_pumping=True,
        period_idx=2,
    )

    assert captured["period_idx"] == 2
    assert out["period_idx"] == 2


def test_compute_indirect_hpr_uses_idx_not_period_id_for_utility_profile(monkeypatch):
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TargetType.II.value] = SimpleNamespace(
        pt=ProblemTable({ProblemTableLabel.T: [120.0, 60.0]})
    )
    calls = {}

    monkeypatch.setattr(hp, "resolve_hpr_target_load", lambda *args, **kwargs: 25.0)
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("profile_kwargs", kwargs)
            or ProblemTable(
                {
                    ProblemTableLabel.T: [120.0, 60.0],
                    ProblemTableLabel.H_NET_HOT: [0.0, -10.0],
                    ProblemTableLabel.H_NET_COLD: [10.0, 0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_targets",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(hp, "_get_hpr_graphs", lambda **kwargs: {})
    monkeypatch.setattr(
        hp,
        "_get_hpr_target_summary",
        lambda res, target_zone: {
            "hpr_cycle": "stub",
            "hpr_utility_total": 11.0,
            "hpr_work": 2.0,
            "hpr_external_utility": 3.0,
            "hpr_ambient_hot": 4.0,
            "hpr_ambient_cold": 5.0,
            "hpr_cop": 6.0,
            "hpr_eta_he": 7.0,
            "hpr_success": True,
            "hpr_hot_streams": StreamCollection(),
            "hpr_cold_streams": StreamCollection(),
            "hpr_details": {},
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: {
            "hot_utilities": StreamCollection(),
            "cold_utilities": StreamCollection(),
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
            "heat_recovery_limit": None,
            "degree_of_int": None,
            "utility_cost": 0.0,
            "hot_pinch": None,
            "cold_pinch": None,
        },
    )
    monkeypatch.setattr(
        hp.IndirectHeatPumpTarget,
        "model_validate",
        classmethod(lambda cls, value: value),
    )

    target_result = hp.compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args={"period_id": "peak", "period_idx": 1},
    )

    assert calls["profile_kwargs"]["period_idx"] == 1
    assert "period_id" not in calls["profile_kwargs"]
    assert target_result["period_id"] == "peak"
    assert target_result["period_idx"] == 1


def test_indirect_hpr_load_uses_finite_utility_profile_when_base_target_has_nans(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.targets[TargetType.II.value] = SimpleNamespace(
        pt=ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET_HOT: [np.nan, np.nan],
                ProblemTableLabel.H_NET_COLD: [np.nan, np.nan],
            }
        )
    )
    utility_profile = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 60.0],
            ProblemTableLabel.H_NET_HOT: [0.0, -40.0],
            ProblemTableLabel.H_NET_COLD: [100.0, 0.0],
        }
    )
    captured = {}

    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: utility_profile,
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_targets",
        lambda **kwargs: (
            captured.__setitem__("target_load", kwargs["Q_hpr_target"])
            or SimpleNamespace()
        ),
    )
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(hp, "_get_hpr_graphs", lambda **kwargs: {})
    monkeypatch.setattr(
        hp,
        "_get_hpr_target_summary",
        lambda res, target_zone: {
            "hpr_cycle": "stub",
            "hpr_utility_total": 11.0,
            "hpr_work": 2.0,
            "hpr_external_utility": 3.0,
            "hpr_ambient_hot": 4.0,
            "hpr_ambient_cold": 5.0,
            "hpr_cop": 6.0,
            "hpr_eta_he": 7.0,
            "hpr_success": True,
            "hpr_hot_streams": StreamCollection(),
            "hpr_cold_streams": StreamCollection(),
            "hpr_details": {},
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: {
            "hot_utilities": StreamCollection(),
            "cold_utilities": StreamCollection(),
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
            "heat_recovery_limit": None,
            "degree_of_int": None,
            "utility_cost": 0.0,
            "hot_pinch": None,
            "cold_pinch": None,
        },
    )
    monkeypatch.setattr(
        hp.IndirectHeatPumpTarget,
        "model_validate",
        classmethod(lambda cls, value: value),
    )

    target_result = hp.compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
    )

    assert target_result["hpr_success"] is True
    assert captured["target_load"] == pytest.approx(100.0)


def test_resolve_hpr_target_load_ignores_nan_load_entries():
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0],
            ProblemTableLabel.H_NET_HOT: [np.nan, -20.0, -40.0],
            ProblemTableLabel.H_NET_COLD: [np.nan, 100.0, 0.0],
        }
    )
    config = Configuration(
        options={"HPR_LOAD_MODE": "fraction", "HPR_LOAD_FRACTION": 0.25}
    )

    assert resolve_hpr_target_load(
        H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
        H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
        is_heat_pumping=True,
        config=config,
    ) == pytest.approx(25.0)


def test_resolve_hpr_target_load_returns_zero_for_all_nan_load_entries():
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0],
            ProblemTableLabel.H_NET_HOT: [np.nan, np.nan],
            ProblemTableLabel.H_NET_COLD: [np.nan, np.nan],
        }
    )

    assert (
        resolve_hpr_target_load(
            H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
            H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
            is_heat_pumping=True,
            config=Configuration(),
        )
        == 0.0
    )


def test_resolve_hpr_target_load_uses_period_load_for_selected_period_id():
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0],
            ProblemTableLabel.H_NET_HOT: [-20.0, -40.0, 0.0],
            ProblemTableLabel.H_NET_COLD: [100.0, 80.0, 0.0],
        }
    )
    config = Configuration(
        options={
            "HPR_LOAD_MODE": "period_values",
            "HPR_LOAD_PERIOD_VALUES": {"base": 10.0, "peak": 25.0},
        }
    )

    assert resolve_hpr_target_load(
        H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
        H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
        is_heat_pumping=True,
        config=config,
        period_id="peak",
        period_idx=1,
    ) == pytest.approx(25.0)


def test_resolve_hpr_target_load_rejects_missing_period_load():
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0],
            ProblemTableLabel.H_NET_HOT: [-20.0, -40.0, 0.0],
            ProblemTableLabel.H_NET_COLD: [100.0, 80.0, 0.0],
        }
    )
    config = Configuration(
        options={
            "HPR_LOAD_MODE": "period_values",
            "HPR_LOAD_PERIOD_VALUES": {"base": 10.0},
        }
    )

    with pytest.raises(ValueError, match="does not define a load"):
        resolve_hpr_target_load(
            H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
            H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
            is_heat_pumping=True,
            config=config,
            period_id="peak",
            period_idx=1,
        )


def test_resolve_hpr_target_load_covers_load_modes_and_invalid_inputs():
    H_net_cold = np.array([0.0, 100.0, 0.0])
    H_net_hot = np.array([0.0, -60.0, 0.0])

    with pytest.raises(ValueError, match="config"):
        resolve_hpr_target_load(
            H_net_cold=H_net_cold,
            H_net_hot=H_net_hot,
            is_heat_pumping=True,
            config=None,
        )

    assert (
        resolve_hpr_target_load(
            H_net_cold=H_net_cold,
            H_net_hot=H_net_hot,
            config=Configuration(),
        )
        == 0
    )
    assert resolve_hpr_target_load(
        H_net_cold=H_net_cold,
        H_net_hot=H_net_hot,
        is_refrigeration=True,
        config=Configuration(options={"HPR_LOAD_MODE": "duty", "HPR_LOAD_DUTY": 80.0}),
    ) == pytest.approx(60.0)
    assert resolve_hpr_target_load(
        H_net_cold=H_net_cold,
        H_net_hot=H_net_hot,
        is_heat_pumping=True,
        config=Configuration(
            options={
                "HPR_LOAD_MODE": "period_values",
                "HPR_LOAD_PERIOD_VALUES": {"1": 12.0},
            }
        ),
        period_id="missing",
        period_idx=1,
    ) == pytest.approx(12.0)

    with pytest.raises(ValueError, match="Unsupported HPR_LOAD_MODE"):
        resolve_hpr_target_load(
            H_net_cold=H_net_cold,
            H_net_hot=H_net_hot,
            is_heat_pumping=True,
            config=SimpleNamespace(
                hpr=SimpleNamespace(load_mode="unsupported"),
            ),
        )


def test_compute_direct_heat_pump_target_orchestrates_target_summary(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TargetType.DI.value] = SimpleNamespace(
        pt=ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET_HOT: [0.0, -40.0],
                ProblemTableLabel.H_NET_COLD: [100.0, 0.0],
            }
        )
    )
    captured = {}
    hpr_result = SimpleNamespace(
        utility_tot=11.0,
        w_net=2.0,
        Q_ext=3.0,
        Q_amb_hot=4.0,
        Q_amb_cold=5.0,
        cop_h=6.0,
        eta_he=7.0,
        hpr_operating_cost=8.0,
        hpr_capital_cost=9.0,
        hpr_annualized_capital_cost=10.0,
        hpr_total_annualized_cost=11.0,
        hpr_compressor_capital_cost=12.0,
        hpr_heat_exchanger_capital_cost=13.0,
        success=True,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
    )

    def fake_get_hpr_targets(**kwargs):
        captured["targets"] = kwargs
        return hpr_result

    def fake_calc_hpr_cascade(**kwargs):
        captured["cascade"] = kwargs
        return ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET_HOT: [0.0, -40.0],
                ProblemTableLabel.H_NET_COLD: [100.0, 0.0],
                ProblemTableLabel.H_HOT_HP: [0.0, 5.0],
                ProblemTableLabel.H_COLD_HP: [3.0, 0.0],
                ProblemTableLabel.H_NET_W_AIR: [4.0, 1.0],
                ProblemTableLabel.H_NET_HP: [2.0, -2.0],
            }
        )

    monkeypatch.setattr(hp, "_get_hpr_targets", fake_get_hpr_targets)
    monkeypatch.setattr(hp, "_calc_hpr_cascade", fake_calc_hpr_cascade)
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **_kwargs: {
            "hot_utilities": StreamCollection(),
            "cold_utilities": StreamCollection(),
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
            "heat_recovery_limit": None,
            "degree_of_int": None,
            "utility_cost": 0.0,
            "hot_pinch": None,
            "cold_pinch": None,
        },
    )
    monkeypatch.setattr(
        hp.DirectHeatPumpTarget,
        "model_validate",
        classmethod(lambda cls, value: value),
    )

    target_result = hp.compute_direct_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args={"period_id": "peak", "period_idx": 1},
    )

    assert captured["targets"]["Q_hpr_target"] == pytest.approx(100.0)
    assert captured["targets"]["period_idx"] == 1
    assert captured["cascade"]["period_idx"] == 1
    assert target_result["type"] == TargetType.DHP.value
    assert target_result["period_id"] == "peak"
    assert target_result["hpr_total_annualized_cost"] == pytest.approx(11.0)


def test_compute_direct_hpr_returns_none_when_direct_profile_has_no_load():
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.targets[TargetType.DI.value] = SimpleNamespace(
        pt=ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET_HOT: [0.0, 0.0],
                ProblemTableLabel.H_NET_COLD: [0.0, 0.0],
            }
        )
    )

    assert (
        hp.compute_direct_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=True,
        )
        is None
    )


def test_compute_indirect_hpr_returns_none_when_utility_profile_has_no_load(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.targets[TargetType.II.value] = SimpleNamespace(
        pt=ProblemTable({ProblemTableLabel.T: [120.0, 60.0]})
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET_HOT: [0.0, 0.0],
                ProblemTableLabel.H_NET_COLD: [0.0, 0.0],
            }
        ),
    )

    assert (
        hp.compute_indirect_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=True,
        )
        is None
    )


def test_hpr_residual_utility_summary_retargets_direct_utilities():
    hot_utilities, cold_utilities = _make_base_utility_collections()
    base_target = SimpleNamespace(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        hot_utility_target=40.0,
        cold_utility_target=30.0,
        heat_recovery_target=10.0,
        heat_recovery_limit=50.0,
    )
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0, 20.0],
            ProblemTableLabel.H_NET_W_AIR: [40.0, 0.0, 0.0, 30.0],
            ProblemTableLabel.H_NET_HP: [15.0, 0.0, 0.0, 10.0],
        }
    )

    summary = hp._get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        period_idx=0,
        is_direct=True,
        is_heat_pumping=True,
    )

    assert summary["hot_utility_target"] == pytest.approx(25.0)
    assert summary["cold_utility_target"] == pytest.approx(20.0)
    assert summary["heat_recovery_target"] == pytest.approx(20.0)
    assert summary["degree_of_int"] == pytest.approx(0.4)
    assert summary["hot_pinch"] == pytest.approx(80.0)
    assert summary["cold_pinch"] == pytest.approx(60.0)
    assert float(summary["hot_utilities"][0].heat_flow[0]) == pytest.approx(25.0)
    assert float(summary["cold_utilities"][0].heat_flow[0]) == pytest.approx(20.0)
    np.testing.assert_allclose(
        pt[ProblemTableLabel.H_NET_HOT_AFTR_HP], np.array([0.0, 0.0, 0.0, -20.0])
    )
    np.testing.assert_allclose(
        pt[ProblemTableLabel.H_NET_COLD_AFTR_HP], np.array([25.0, 0.0, 0.0, 0.0])
    )


def test_hpr_residual_utility_summary_removes_direct_hpr_pockets():
    hot_utilities, cold_utilities = _make_base_utility_collections()
    base_target = SimpleNamespace(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        hot_utility_target=40.0,
        cold_utility_target=30.0,
        heat_recovery_target=10.0,
        heat_recovery_limit=50.0,
    )
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0, 20.0],
            ProblemTableLabel.H_NET_W_AIR: [10.0, 500.0, 0.0, 0.0],
            ProblemTableLabel.H_NET_HP: [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = hp._get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        period_idx=0,
        is_direct=True,
        is_heat_pumping=True,
    )

    assert summary["hot_utility_target"] == pytest.approx(10.0)
    assert summary["cold_utility_target"] == pytest.approx(0.0)
    assert summary["heat_recovery_target"] == pytest.approx(40.0)
    assert float(summary["hot_utilities"][0].heat_flow[0]) == pytest.approx(10.0)


def test_hpr_residual_utility_summary_retargets_indirect_utilities():
    hot_utilities, cold_utilities = _make_base_utility_collections()
    base_target = SimpleNamespace(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        hot_utility_target=40.0,
        cold_utility_target=30.0,
        heat_recovery_target=10.0,
        heat_recovery_limit=50.0,
    )
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0, 60.0, 20.0],
            ProblemTableLabel.H_NET_UT: [40.0, 0.0, 0.0, 30.0],
            ProblemTableLabel.H_NET_HP: [15.0, 0.0, 0.0, 10.0],
            ProblemTableLabel.RCP_UT_NET: [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = hp._get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        period_idx=0,
        is_direct=False,
        is_heat_pumping=True,
    )

    assert summary["hot_utility_target"] == pytest.approx(25.0)
    assert summary["cold_utility_target"] == pytest.approx(20.0)
    assert summary["heat_recovery_target"] == pytest.approx(25.0)
    assert summary["degree_of_int"] == pytest.approx(0.5)
    assert summary["hot_pinch"] == pytest.approx(80.0)
    assert summary["cold_pinch"] == pytest.approx(60.0)
    assert float(summary["hot_utilities"][0].heat_flow[0]) == pytest.approx(25.0)
    assert float(summary["cold_utilities"][0].heat_flow[0]) == pytest.approx(20.0)
    np.testing.assert_allclose(
        pt[ProblemTableLabel.H_NET_HOT_UT_AFTR_HP], np.array([25.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        pt[ProblemTableLabel.H_NET_COLD_UT_AFTR_HP], np.array([0.0, 0.0, 0.0, -20.0])
    )


def test_plot_multi_hp_profiles_from_results_returns_plotly_figure():
    hot_streams = StreamCollection()
    hot_streams.add(
        Stream(
            name="HP_H1",
            supply_temperature=110.0,
            target_temperature=100.0,
            heat_flow=20.0,
        )
    )
    cold_streams = StreamCollection()
    cold_streams.add(
        Stream(
            name="HP_C1",
            supply_temperature=70.0,
            target_temperature=80.0,
            heat_flow=15.0,
        )
    )

    figure = hp_plotting.plot_multi_hp_profiles_from_results(
        T_hot=np.array([120.0, 100.0]),
        H_hot=np.array([0.0, 20.0]),
        T_cold=np.array([80.0, 60.0]),
        H_cold=np.array([0.0, 15.0]),
        hpr_hot_streams=hot_streams,
        hpr_cold_streams=cold_streams,
        period_idx=0,
        title="HP Profile",
    )

    assert figure.layout.title.text == "HP Profile"
    assert [trace.name for trace in figure.data] == [
        "Sink",
        "Source",
        "Condenser",
        "Evaporator",
    ]


def test_get_hpr_graphs_covers_refrigeration_and_indirect_heat_pump_outputs():
    refrigeration_graphs = hp._get_hpr_graphs(
        ProblemTable({ProblemTableLabel.T: [120.0, 80.0]}),
        is_direct=True,
        is_heat_pumping=False,
    )
    indirect_pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0],
            ProblemTableLabel.H_NET_UT: [4.0, 0.0],
            ProblemTableLabel.H_NET_HP: [2.0, -2.0],
        }
    )
    indirect_graphs = hp._get_hpr_graphs(
        indirect_pt,
        is_direct=False,
        is_heat_pumping=True,
    )

    assert refrigeration_graphs == {}
    assert set(indirect_graphs) == {GraphType.SUGCC.value}
    assert list(indirect_graphs[GraphType.SUGCC.value].columns) == [
        ProblemTableLabel.T.value,
        ProblemTableLabel.H_NET_UT.value,
        ProblemTableLabel.H_NET_HP.value,
    ]


def test_direct_heat_pump_graphs_include_nlp_and_hpr_overlay():
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 80.0],
            ProblemTableLabel.H_NET_HOT: [8.0, 2.0],
            ProblemTableLabel.H_NET_COLD: [1.0, 6.0],
            ProblemTableLabel.H_HOT_UT: [0.0, 3.0],
            ProblemTableLabel.H_COLD_UT: [4.0, 0.0],
            ProblemTableLabel.H_HOT_HP: [0.0, 5.0],
            ProblemTableLabel.H_COLD_HP: [3.0, 0.0],
            ProblemTableLabel.H_NET_W_AIR: [4.0, 1.0],
            ProblemTableLabel.H_NET_HP: [2.0, -2.0],
        }
    )

    graphs = hp._get_hpr_graphs(pt, is_direct=True, is_heat_pumping=True)

    assert set(graphs) == {GraphType.NLP_HP.value, GraphType.GCC_HP.value}
    assert list(graphs[GraphType.NLP_HP.value].columns) == [
        ProblemTableLabel.T.value,
        ProblemTableLabel.H_NET_HOT.value,
        ProblemTableLabel.H_NET_COLD.value,
        ProblemTableLabel.H_HOT_HP.value,
        ProblemTableLabel.H_COLD_HP.value,
    ]

    graph_set = _create_graph_set(
        SimpleNamespace(
            name="Direct Heat Pump",
            type="Direct Heat Pump",
            graphs=graphs,
        )
    )
    nlp_hp_graph = next(
        graph
        for graph in graph_set["graphs"]
        if graph["type"] == GraphType.NLP_HP.value
    )
    hpr_segment_titles = {segment["title"] for segment in nlp_hp_graph["segments"]}
    assert "Heat Pump Condenser" in hpr_segment_titles
    assert "Heat Pump Evaporator" in hpr_segment_titles


@pytest.mark.parametrize(
    "hpr_type",
    [
        HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
        HeatPumpAndRefrigerationCycle.ParallelCarnot.value,
        HeatPumpAndRefrigerationCycle.ParallelVapourComp.value,
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
    ],
)
def test_get_hpr_targets_validates_supported_non_brayton_backend_results(
    monkeypatch, hpr_type
):
    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda **kwargs: SimpleNamespace(hpr_type=hpr_type),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        hpr_type,
        lambda args: HPRBackendResult(
            obj=0.1,
            utility_tot=1.0,
            w_net=0.5,
            Q_ext_heat=0.25,
            Q_ext_cold=0.25,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
            amb_streams=StreamCollection(),
        ),
    )

    out = hp._get_hpr_targets(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        config=SimpleNamespace(HPR_TYPE=hpr_type),
        is_heat_pumping=True,
    )

    assert isinstance(out, hp.HeatPumpTargetOutputs)
    assert out.success is True
    assert out.Q_ext == pytest.approx(0.5)


def test_get_hpr_targets_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda **_kwargs: SimpleNamespace(hpr_type="not-a-backend"),
    )

    with pytest.raises(ValueError, match="No valid heat pump targeting type"):
        hp._get_hpr_targets(
            Q_hpr_target=10.0,
            T_vals=np.array([120.0, 80.0]),
            H_hot=np.array([0.0, -10.0]),
            H_cold=np.array([10.0, 0.0]),
            config=Configuration(),
            is_heat_pumping=True,
        )


@pytest.mark.parametrize(
    ("failing_helper", "attribute"),
    [
        ("grid", "create_problem_table_with_t_int"),
        ("air", "get_process_heat_cascade"),
        ("utility", "get_utility_heat_cascade"),
    ],
)
def test_calc_hpr_cascade_propagates_helper_type_error_once(
    monkeypatch, failing_helper, attribute
):
    calls = {"grid": 0, "air": 0, "utility": 0}

    def fake_problem_table_grid(**_kwargs):
        calls["grid"] += 1
        return ProblemTable({ProblemTableLabel.T: [120.0, 60.0]})

    def fake_process_heat_cascade(**_kwargs):
        calls["air"] += 1
        return ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 60.0],
                ProblemTableLabel.H_NET: [1.0, 1.0],
                ProblemTableLabel.H_NET_HOT: [0.0, 0.0],
                ProblemTableLabel.H_NET_COLD: [1.0, 1.0],
            }
        )

    def fake_utility_heat_cascade(**kwargs):
        calls["utility"] += 1
        return {
            "T_col": np.array(kwargs["T_int_vals"], dtype=float),
            "updates": {
                ProblemTableLabel.H_NET_UT: np.array([3.0, -3.0]),
                ProblemTableLabel.H_HOT_UT: np.array([0.0, -3.0]),
                ProblemTableLabel.H_COLD_UT: np.array([3.0, 0.0]),
            },
        }

    helpers = {
        "grid": fake_problem_table_grid,
        "air": fake_process_heat_cascade,
        "utility": fake_utility_heat_cascade,
    }

    def fail_once(**_kwargs):
        calls[failing_helper] += 1
        raise TypeError("internal helper failure")

    monkeypatch.setattr(hp, "create_problem_table_with_t_int", helpers["grid"])
    monkeypatch.setattr(hp, "get_process_heat_cascade", helpers["air"])
    monkeypatch.setattr(hp, "get_utility_heat_cascade", helpers["utility"])
    monkeypatch.setattr(hp, attribute, fail_once)

    ambient_streams = StreamCollection()
    ambient_streams.add(
        Stream(
            name="Air", supply_temperature=50.0, target_temperature=80.0, heat_flow=10.0
        )
    )
    pt = ProblemTable(
        {
            ProblemTableLabel.T: [120.0, 60.0],
            ProblemTableLabel.H_NET_A: [2.0, 2.0],
            ProblemTableLabel.H_NET_HOT: [1.0, 1.0],
            ProblemTableLabel.H_NET_COLD: [2.0, 2.0],
        }
    )
    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        amb_streams=ambient_streams,
    )

    with pytest.raises(TypeError, match="internal helper failure"):
        hp._calc_hpr_cascade(
            pt,
            res,
            is_T_vals_shifted=True,
            is_heat_pumping=False,
            period_idx=5,
        )

    assert calls[failing_helper] == 1


def test_hpr_handler_registry_includes_brayton():
    handler = hp._HP_PLACEMENT_HANDLERS[HeatPumpAndRefrigerationCycle.Brayton.value]
    assert handler is hp.optimise_brayton_heat_pump_placement
