"""Multi-period heat pump and refrigeration optimisation tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps._multiperiod.aggregation as hp_aggregation
import OpenPinch.analysis.heat_pumps._multiperiod.execution as hp_execution
import OpenPinch.analysis.heat_pumps._multiperiod.preparation as hp_preparation
import OpenPinch.analysis.heat_pumps.service as hp
import OpenPinch.application.targeting as svc
from OpenPinch.analysis.heat_pumps._multiperiod.state import (
    _PreparedHPRPeriodCase,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.hpr import (
    HeatPumpTargetInputs,
    HeatPumpTargetOutputs,
    HPRBackendResult,
    HPRPeriodCase,
    HPRThermoArtifacts,
    MultiPeriodHPRTargetInputs,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import PT, TT, HPRcycle
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import DirectHeatPumpTarget
from OpenPinch.domain.value import Value
from OpenPinch.domain.zone import Zone
from OpenPinch.optimisation.models import OptimisationCandidate

from .helpers import _base_args


def _input_args(**overrides) -> HeatPumpTargetInputs:
    data = vars(_base_args(**overrides)).copy()
    data.setdefault("n_mvr", 1)
    data.setdefault("eta_mvr_comp", 0.7)
    data.setdefault("eta_motor", 0.95)
    data.setdefault("mvr_fluid_ls", ["Water"])
    return HeatPumpTargetInputs.model_validate(data)


def _backend_result(*, obj: float, utility_tot: float, period_idx: int):
    return HPRBackendResult(
        obj=obj,
        utility_tot=utility_tot,
        w_net=utility_tot / 10.0,
        Q_ext_heat=1.0 + period_idx,
        Q_ext_cold=2.0 + period_idx,
        hpr_operating_cost=Value(100.0 + 10.0 * period_idx, "$/y"),
        hpr_capital_cost=Value(1000.0 + 100.0 * period_idx, "$"),
        hpr_annualized_capital_cost=Value(200.0 + 20.0 * period_idx, "$/y"),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        cop_h=3.0 + period_idx,
        amb_streams=StreamCollection(),
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
    )


def _hpr_output(**updates) -> HeatPumpTargetOutputs:
    payload = {
        "utility_tot": 1.0,
        "w_net": 1.0,
        "Q_ext": 0.0,
        "Q_amb_hot": 0.0,
        "Q_amb_cold": 0.0,
        "cop_h": 3.0,
        "eta_he": 0.0,
        "obj": 0.0,
        "success": True,
        "hpr_hot_streams": StreamCollection(),
        "hpr_cold_streams": StreamCollection(),
        "amb_streams": StreamCollection(),
    }
    payload.update(updates)
    return HeatPumpTargetOutputs.model_validate(payload)


def _pt(T, H_cold, H_hot):
    return ProblemTable(
        {
            PT.T: T,
            PT.H_NET: np.asarray(H_cold) + np.asarray(H_hot),
            PT.H_NET_COLD: H_cold,
            PT.H_NET_HOT: H_hot,
            PT.H_NET_A: np.asarray(H_cold) + np.asarray(H_hot),
        }
    )


def _base_target(pt):
    return SimpleNamespace(
        pt=pt,
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
        heat_recovery_limit=100.0,
    )


def _residual_summary():
    return {
        "hot_utilities": StreamCollection(),
        "cold_utilities": StreamCollection(),
        "hot_utility_target": 0.0,
        "cold_utility_target": 0.0,
        "heat_recovery_target": 0.0,
        "heat_recovery_limit": 100.0,
        "degree_of_int": 0.0,
        "utility_cost": 0.0,
        "hot_pinch": None,
        "cold_pinch": None,
    }


def test_solve_hpr_multiperiod_placement_scores_weighted_shared_vector(
    monkeypatch,
):
    monkeypatch.setattr(
        hp_execution,
        "run_hpr_candidate_search",
        lambda **_kwargs: (
            OptimisationCandidate(objective=427.5, point=(3.0,)),
            OptimisationCandidate(objective=627.5, point=(1.0,)),
        ),
    )
    cases = [
        HPRPeriodCase(
            period_id="p0",
            period_idx=0,
            weight=1.0,
            args=_input_args(period_idx=0),
        ),
        HPRPeriodCase(
            period_id="p1",
            period_idx=1,
            weight=3.0,
            args=_input_args(period_idx=1),
        ),
    ]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HPRcycle.CascadeCarnot.value,
        max_multi_start=2,
        bb_minimiser="rbf",
    )

    def objective(x, args, debug=False):
        target = 1.0 if args.period_idx == 0 else 3.0
        obj = float((x[0] - target) ** 2)
        result = _backend_result(
            obj=obj,
            utility_tot=float(x[0] + 10.0 * args.period_idx),
            period_idx=args.period_idx,
        )
        return result.with_updates(
            hpr_operating_cost=Value(
                float(result.hpr_operating_cost.value) + 100.0 * obj,
                "$/y",
            )
        )

    result = hp_execution.solve_hpr_multiperiod_placement(
        f_obj=objective,
        x0_ls=None,
        bnds=[(0.0, 4.0)],
        args=mp_args,
    )

    assert result.obj == pytest.approx(427.5)
    assert result.utility_tot == pytest.approx(3.0)
    np.testing.assert_allclose(result.design_vector, np.array([3.0]))
    assert result.period_ids == ["p0", "p1"]
    assert result.period_weights == [1.0, 3.0]
    assert set(result.period_outputs) == {"p0", "p1"}
    assert result.weighted_output.utility_tot == pytest.approx(10.5)
    assert result.weighted_output.hpr_operating_cost.value == pytest.approx(207.5)
    assert result.weighted_output.hpr_capital_cost.value == pytest.approx(1100.0)
    assert result.weighted_output.hpr_total_annualized_cost.value == pytest.approx(
        427.5
    )


def test_shared_hpr_candidate_score_uses_weighted_operation_and_peak_capital(
    monkeypatch,
):
    monkeypatch.setattr(
        hp_execution,
        "run_hpr_candidate_search",
        lambda **_kwargs: (
            OptimisationCandidate(objective=385.0, point=(2.0,)),
            OptimisationCandidate(objective=750.0, point=(1.0,)),
        ),
    )
    cases = [
        HPRPeriodCase(
            period_id="p0",
            period_idx=0,
            weight=1.0,
            args=_input_args(period_idx=0),
        ),
        HPRPeriodCase(
            period_id="p1",
            period_idx=1,
            weight=3.0,
            args=_input_args(period_idx=1),
        ),
    ]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HPRcycle.CascadeCarnot.value,
        max_multi_start=2,
        bb_minimiser="rbf",
    )

    def objective(x, args, debug=False):
        candidate = int(x[0])
        period_idx = int(args.period_idx)
        operating = {
            (1, 0): 100.0,
            (1, 1): 300.0,
            (2, 0): 400.0,
            (2, 1): 100.0,
        }[candidate, period_idx]
        annualized_capital = {
            (1, 0): 500.0,
            (1, 1): 50.0,
            (2, 0): 100.0,
            (2, 1): 200.0,
        }[candidate, period_idx]
        penalty = 40.0 if candidate == 2 and period_idx == 0 else 0.0
        return _backend_result(
            obj=1.0 if candidate == 1 else 100.0,
            utility_tot=float(candidate),
            period_idx=period_idx,
        ).with_updates(
            hpr_operating_cost=Value(operating, "$/y"),
            hpr_annualized_capital_cost=Value(annualized_capital, "$/y"),
            feasibility_penalty=penalty,
        )

    result = hp_execution.solve_hpr_multiperiod_placement(
        f_obj=objective,
        x0_ls=None,
        bnds=[(0.0, 3.0)],
        args=mp_args,
    )

    np.testing.assert_allclose(result.design_vector, np.array([2.0]))
    assert result.obj == pytest.approx(385.0)
    assert result.weighted_output.hpr_operating_cost.value == pytest.approx(175.0)
    assert result.weighted_output.feasibility_penalty == pytest.approx(10.0)
    assert result.weighted_output.hpr_annualized_capital_cost.value == pytest.approx(
        200.0
    )
    assert result.weighted_output.hpr_total_annualized_cost.value == pytest.approx(
        375.0
    )


def test_shared_hpr_candidate_score_falls_back_when_cost_breakdown_absent():
    cases = [
        HPRPeriodCase(
            period_id="p0",
            period_idx=0,
            weight=1.0,
            args=_input_args(period_idx=0),
        ),
        HPRPeriodCase(
            period_id="p1",
            period_idx=1,
            weight=3.0,
            args=_input_args(period_idx=1),
        ),
    ]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HPRcycle.CascadeCarnot.value,
        max_multi_start=1,
        bb_minimiser="rbf",
    )

    def objective(x, args, debug=False):
        return HPRBackendResult(
            obj=10.0 + 10.0 * args.period_idx,
            utility_tot=1.0,
            w_net=1.0,
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
        )

    result = hp_aggregation.evaluate_multiperiod_candidate(
        np.array([1.0]),
        mp_args,
        period_objective=objective,
    )

    assert result.obj == pytest.approx(17.5)


def test_solve_hpr_multiperiod_placement_fails_when_any_period_fails(monkeypatch):
    monkeypatch.setattr(
        hp_execution,
        "run_hpr_candidate_search",
        lambda **_kwargs: (OptimisationCandidate(objective=1e30, point=(1.0,)),),
    )
    cases = [
        HPRPeriodCase(
            period_id="p0",
            period_idx=0,
            weight=1.0,
            args=_input_args(period_idx=0),
        ),
        HPRPeriodCase(
            period_id="p1",
            period_idx=1,
            weight=1.0,
            args=_input_args(period_idx=1),
        ),
    ]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HPRcycle.CascadeCarnot.value,
        max_multi_start=1,
        bb_minimiser="rbf",
    )

    def objective(x, args, debug=False):
        if args.period_idx == 1:
            return HPRBackendResult.failure(reason="period solve failed")
        return _backend_result(obj=0.0, utility_tot=1.0, period_idx=args.period_idx)

    with pytest.raises(ValueError, match="failed to return an optimal result"):
        hp_execution.solve_hpr_multiperiod_placement(
            f_obj=objective,
            x0_ls=None,
            bnds=[(0.0, 1.0)],
            args=mp_args,
        )


def test_build_multiperiod_cases_aligns_period_temperature_grids(monkeypatch):
    zone = Zone(
        config=Configuration(
            options={
                "PROBLEM_PERIOD_IDS": ["p0", "p1"],
                "PROBLEM_PERIOD_WEIGHTS": [1.0, 3.0],
            }
        )
    )
    base_by_period = {
        "p0": _base_target(_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0])),
        "p1": _base_target(
            _pt([140.0, 90.0, 60.0], [130.0, 40.0, 0.0], [0.0, -20.0, -80.0])
        ),
    }
    monkeypatch.setattr(
        hp_preparation,
        "_compute_hpr_base_target_for_period",
        lambda *, zone, period_args, is_direct: base_by_period[
            period_args["period_id"]
        ],
    )

    cases = hp_preparation.build_multiperiod_hpr_cases(
        zone=zone,
        is_heat_pumping=True,
        is_direct=True,
    )

    assert [case.period_id for case in cases] == ["p0", "p1"]
    assert [case.weight for case in cases] == [1.0, 3.0]
    np.testing.assert_allclose(cases[0].optimizer_pt[PT.T], cases[1].optimizer_pt[PT.T])
    np.testing.assert_allclose(
        cases[0].optimizer_pt[PT.T],
        np.array([140.0, 120.0, 90.0, 60.0]),
    )


@pytest.mark.parametrize(
    "cycle",
    [
        HPRcycle.CascadeCarnot.value,
        HPRcycle.ParallelCarnot.value,
        HPRcycle.CascadeVapourComp.value,
        HPRcycle.ParallelVapourComp.value,
        HPRcycle.VapourCompMVR.value,
    ],
)
def test_supported_hpr_cycles_prepare_shared_target_inputs(
    monkeypatch,
    cycle,
):
    cases = [
        _PreparedHPRPeriodCase(
            period_id="p0",
            period_idx=0,
            weight=1.0,
            solver_case=HPRPeriodCase(
                period_id="p0",
                period_idx=0,
                weight=1.0,
                args=_input_args(
                    hpr_type=cycle,
                    initialise_simulated_cycle=False,
                    n_cond=2,
                    n_evap=2,
                ),
            ),
            base_target=_base_target(_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0])),
            optimizer_pt=_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0]),
        ),
        _PreparedHPRPeriodCase(
            period_id="p1",
            period_idx=1,
            weight=1.0,
            solver_case=HPRPeriodCase(
                period_id="p1",
                period_idx=1,
                weight=1.0,
                args=_input_args(
                    hpr_type=cycle,
                    initialise_simulated_cycle=False,
                    n_cond=2,
                    n_evap=2,
                    period_idx=1,
                ),
            ),
            base_target=_base_target(_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0])),
            optimizer_pt=_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0]),
        ),
    ]
    captured = {}

    def fake_solve_hpr_multiperiod_placement(*, f_obj, x0_ls, bnds, args):
        captured["period_ids"] = [case.period_id for case in args.period_cases]
        captured["hpr_type"] = args.hpr_type
        captured["has_bounds"] = bool(bnds)
        captured["has_initial_candidate"] = x0_ls is not None
        return _backend_result(obj=1.0, utility_tot=2.0, period_idx=0)

    monkeypatch.setattr(
        hp_execution,
        "solve_hpr_multiperiod_placement",
        fake_solve_hpr_multiperiod_placement,
    )

    result = hp_execution.get_multiperiod_hpr_targets(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
    )

    assert result.success is True
    assert result.utility_tot == pytest.approx(2.0)
    assert captured["period_ids"] == ["p0", "p1"]
    assert captured["hpr_type"] == cycle
    assert captured["has_bounds"] is True
    if cycle in {HPRcycle.CascadeCarnot.value, HPRcycle.ParallelCarnot.value}:
        assert captured["has_initial_candidate"] is True


def test_hpr_multiperiod_flag_false_uses_selected_period_path(monkeypatch):
    zone = Zone(
        config=Configuration(
            options={
                "PROBLEM_PERIOD_IDS": ["p0", "p1"],
                "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": False,
            }
        )
    )
    zone.targets[TT.DI.value] = _base_target(
        _pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0])
    )
    calls = {"selected": 0}

    monkeypatch.setattr(
        hp,
        "_compute_multiperiod_heat_pump_or_refrigeration_target",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    def fake_get_hpr_targets(**kwargs):
        calls["selected"] += 1
        assert kwargs["period_idx"] == 1
        return _hpr_output(utility_tot=11.0, w_net=2.0)

    monkeypatch.setattr(hp, "_get_hpr_targets", fake_get_hpr_targets)
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: _residual_summary(),
    )

    target = hp.compute_direct_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args={"period_id": "p1", "period_idx": 1},
    )

    assert calls["selected"] == 1
    assert target.period_id == "p1"
    assert target.hpr_utility_total == pytest.approx(11.0)


def test_hpr_multiperiod_flag_true_returns_selected_period_from_shared_design(
    monkeypatch,
):
    zone = Zone(
        config=Configuration(
            options={
                "PROBLEM_PERIOD_IDS": ["p0", "p1"],
                "PROBLEM_PERIOD_WEIGHTS": [1.0, 3.0],
                "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True,
            }
        )
    )
    base_by_period = {
        "p0": _base_target(_pt([120.0, 60.0], [100.0, 0.0], [0.0, -50.0])),
        "p1": _base_target(
            _pt([140.0, 90.0, 60.0], [130.0, 40.0, 0.0], [0.0, -20.0, -80.0])
        ),
    }
    captured = {}

    monkeypatch.setattr(
        hp_preparation,
        "_compute_hpr_base_target_for_period",
        lambda *, zone, period_args, is_direct: base_by_period[
            period_args["period_id"]
        ],
    )

    def fake_multiperiod_targets(**kwargs):
        cases = kwargs["period_cases"]
        captured["period_ids"] = [case.period_id for case in cases]
        captured["weights"] = [case.weight for case in cases]
        captured["row_counts"] = [len(case.optimizer_pt[PT.T]) for case in cases]
        return _hpr_output(
            utility_tot=42.0,
            w_net=4.2,
            period_ids=[case.period_id for case in cases],
            period_weights=[case.weight for case in cases],
            period_outputs={"p0": "period0", "p1": "period1"},
            weighted_output="weighted",
        )

    monkeypatch.setattr(hp, "get_multiperiod_hpr_targets", fake_multiperiod_targets)
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: _residual_summary(),
    )

    target = hp.compute_direct_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args={"period_id": "p1", "period_idx": 1},
    )

    assert captured["period_ids"] == ["p0", "p1"]
    assert captured["weights"] == [1.0, 3.0]
    assert len(set(captured["row_counts"])) == 1
    assert target.period_id == "p1"
    assert target.hpr_utility_total == pytest.approx(42.0)
    assert target.hpr_details.period_outputs == {"p0": "period0", "p1": "period1"}


def test_weighted_hpr_summary_uses_shared_design_period_evaluations(monkeypatch):
    payload = {
        "streams": [
            {
                "zone": "Site/AreaA",
                "name": "HotA",
                "t_supply": {"values": [200.0, 220.0], "unit": "degC"},
                "t_target": {"values": [100.0, 120.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 200.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/AreaA",
                "name": "ColdA",
                "t_supply": {"values": [50.0, 60.0], "unit": "degC"},
                "t_target": {"values": [150.0, 170.0], "unit": "degC"},
                "heat_flow": {"values": [80.0, 120.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
        ],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        },
        "options": {
            "PROBLEM_PERIOD_IDS": ["p0", "p1"],
            "PROBLEM_PERIOD_WEIGHTS": [1.0, 3.0],
            "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True,
        },
    }

    def fake_direct_hpr(target_zone, is_heat_pumping, args=None):
        idx = args["period_idx"]
        hpr_utility_total = [10.0, 30.0][idx]
        return DirectHeatPumpTarget(
            zone_name=target_zone.name,
            period_id=args.get("period_id"),
            period_idx=idx,
            type=TT.DHP.value,
            parent_zone=target_zone.parent_zone,
            config=target_zone.config,
            pt=ProblemTable({PT.T: [120.0, 60.0]}),
            hpr_cycle="shared",
            hpr_utility_total=hpr_utility_total,
            hpr_work=1.0 + idx,
            hpr_external_utility=2.0 + idx,
            hpr_ambient_hot=0.0,
            hpr_ambient_cold=0.0,
            hpr_cop=3.0 + idx,
            hpr_eta_he=0.0,
            hpr_success=True,
            hpr_hot_streams=StreamCollection(),
            hpr_cold_streams=StreamCollection(),
            hpr_details={
                "shared_design": True,
                "period_outputs": ["p0", "p1"],
            },
        )

    monkeypatch.setattr(
        svc,
        "compute_direct_heat_pump_or_refrigeration_target",
        fake_direct_hpr,
    )

    problem = PinchProblem(source=payload, project_name="Site")
    problem.target.direct_heat_pump(zone_name="AreaA", period_id="p0")

    frame = problem.summary_frame(detailed=True, periods="weighted_average")
    hpr_rows = frame[frame["HPR Utility Total (value)"].notna()]

    assert len(hpr_rows) == 1
    assert hpr_rows.iloc[0]["Period ID"] == "weighted_average"
    assert hpr_rows.iloc[0]["HPR Utility Total (value)"] == pytest.approx(25.0)
