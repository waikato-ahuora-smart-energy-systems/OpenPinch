"""A shared multiperiod machine keeps a minimum lift in every period.

Each stage's lift in every period is at least HPR_TURNDOWN times its design
lift, the largest lift it runs at in any period. A period that needs less
lift runs at the floor, with its condenser raised: larger approaches and a
worse COP are the turndown cost.
"""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps._multiperiod.execution as hp_execution
import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
import OpenPinch.analysis.heat_pumps.targeting.parallel_vapour_compression as hp_parallel
import OpenPinch.analysis.heat_pumps.targeting.vapour_compression_mvr as hp_mvr
from OpenPinch.analysis.heat_pumps._multiperiod.aggregation import (
    evaluate_multiperiod_candidate,
)
from OpenPinch.analysis.heat_pumps.common.shared import (
    raise_cascade_lifts_to_floor,
    raise_parallel_lifts_to_floor,
)
from OpenPinch.contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPRPeriodCase,
    HPRTargetingError,
    MultiPeriodHPRTargetInputs,
)
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle

from .helpers import _base_args


def test_a_parallel_machine_below_its_floor_condenses_higher():
    T_cond, T_evap = raise_parallel_lifts_to_floor(
        np.array([60.0, 40.0]), np.array([55.0, 10.0]), np.array([20.0, 20.0])
    )

    np.testing.assert_allclose(T_cond, [75.0, 40.0])
    np.testing.assert_allclose(T_evap, [55.0, 10.0])


def test_parallel_condensers_stay_in_descending_order():
    T_cond, _ = raise_parallel_lifts_to_floor(
        np.array([60.0, 58.0]), np.array([20.0, 50.0]), np.array([0.0, 20.0])
    )

    np.testing.assert_allclose(T_cond, [70.0, 70.0])


def test_each_cascade_stage_below_its_floor_is_raised_from_the_bottom_up():
    # Levels 80, 78 | 5: the top stage lifts 2 K, the bottom stage 73 K.
    T_cond, T_evap = raise_cascade_lifts_to_floor(
        np.array([80.0, 78.0]), np.array([5.0]), np.array([10.0, 10.0])
    )

    np.testing.assert_allclose(T_cond, [88.0, 78.0])
    np.testing.assert_allclose(T_evap, [5.0])


def _args(**overrides):
    return _base_args(
        n_cond=1,
        n_evap=1,
        refrigerant_ls=["Ammonia"],
        T_cold=np.array([130.0, 80.0, 30.0]),
        T_hot=np.array([120.0, 70.0, 20.0]),
        **overrides,
    )


def _decode_with_and_without_a_floor(parse, layout_size, lifts_of):
    args = _args()
    x = np.full(layout_size(args), 0.5)
    plain = parse(x, args)
    lift = lifts_of(plain, args)[0]
    args.stage_lift_floor = np.array([lift + 5.0])
    floored = parse(x, args)
    return plain, floored, lift


def test_a_single_stage_decodes_at_least_at_its_floor():
    plain, floored, lift = _decode_with_and_without_a_floor(
        hp_cascade._parse_cascade_hp_state_variables,
        lambda args: len(hp_cascade._get_cascade_hp_opt_setup(None, args)[1]),
        lambda state, args: state.T_cond - state.T_evap,
    )

    assert floored.T_cond[0] - floored.T_evap[0] == pytest.approx(lift + 5.0)
    np.testing.assert_allclose(floored.T_evap, plain.T_evap)


def test_a_parallel_machine_decodes_at_least_at_its_floor():
    plain, floored, lift = _decode_with_and_without_a_floor(
        hp_parallel._parse_parallel_hp_state_temperatures,
        lambda args: hp_parallel._parallel_layout(args).size,
        lambda state, args: state.T_cond - state.T_evap,
    )

    assert floored.T_cond[0] - floored.T_evap[0] == pytest.approx(lift + 5.0)
    np.testing.assert_allclose(floored.T_evap, plain.T_evap)


def test_a_vc_mvr_closed_stage_decodes_at_least_at_its_floor():
    def vc_lifts(state, args):
        parts = hp_mvr._unpack_vc_mvr_state(state, args)
        return np.asarray(parts["T_cond_vc"]) - np.asarray(parts["T_evap_vc"])

    plain, floored, lift = _decode_with_and_without_a_floor(
        hp_mvr._parse_vc_mvr_state_variables,
        lambda args: hp_mvr._vc_mvr_layout(args).size,
        vc_lifts,
    )

    assert vc_lifts(floored, _args())[0] == pytest.approx(lift + 5.0)


_DECODED_LIFTS = {0: 40.0, 1: 4.0}


def _period_case(idx):
    data = vars(_base_args(period_idx=idx, turndown=0.25)).copy()
    data.update(n_mvr=1, eta_mvr_comp=0.7, eta_mvr_motor=0.95, mvr_fluid_ls=[])
    return HPRPeriodCase(
        period_id=f"p{idx}",
        period_idx=idx,
        weight=1.0,
        args=HeatPumpTargetInputs.model_validate(data),
    )


def test_every_period_runs_at_least_at_the_turndown_share_of_the_design_lift():
    cases = [_period_case(0), _period_case(1)]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        max_multi_start=1,
        bb_minimiser="rbf_surrogate",
    )
    floors_seen = {}

    def period_objective(x, args, debug=False):
        floors_seen[args.period_idx] = np.asarray(args.stage_lift_floor).tolist()
        return HPRBackendResult(
            obj=1.0,
            utility_tot=1.0,
            w_net=1.0,
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
        )

    def stage_lifts(x, args):
        # Lifts are read from the plain decode, with no floor applied.
        assert args.stage_lift_floor is None
        return np.array([_DECODED_LIFTS[args.period_idx]])

    evaluate_multiperiod_candidate(
        np.array([0.5]),
        mp_args,
        period_objective=period_objective,
        stage_lifts=stage_lifts,
    )

    # Design lift 40 K: every period lifts at least 0.25 x 40 = 10 K.
    assert floors_seen == {0: [10.0], 1: [10.0]}


def test_the_turndown_is_a_setting():
    assert CONFIG_FIELD_SPECS["HPR_TURNDOWN"].default == 0.25


def test_each_topology_reads_its_stage_lifts_from_the_plain_decode():
    args = _base_args(
        n_cond=2,
        n_evap=1,
        refrigerant_ls=["Ammonia", "Ammonia"],
        T_cold=np.array([130.0, 80.0, 30.0]),
        T_hot=np.array([120.0, 70.0, 20.0]),
    )
    x = np.full(len(hp_cascade._get_cascade_hp_opt_setup(None, args)[1]), 0.3)
    state = hp_cascade._parse_cascade_hp_state_variables(x, args)
    levels = np.concatenate([state.T_cond, state.T_evap])

    np.testing.assert_allclose(
        hp_cascade._cascade_stage_lifts(x, args), levels[:-1] - levels[1:]
    )

    args = _args()
    x = np.full(hp_parallel._parallel_layout(args).size, 0.3)
    state = hp_parallel._parse_parallel_hp_state_temperatures(x, args)
    np.testing.assert_allclose(
        hp_parallel._parallel_stage_lifts(x, args), state.T_cond - state.T_evap
    )

    x = np.full(hp_mvr._vc_mvr_layout(args).size, 0.3)
    parts = hp_mvr._unpack_vc_mvr_state(
        hp_mvr._parse_vc_mvr_state_variables(x, args), args
    )
    np.testing.assert_allclose(
        hp_mvr._vc_mvr_stage_lifts(x, args),
        np.asarray(parts["T_cond_vc"]) - np.asarray(parts["T_evap_vc"]),
    )


def test_the_shared_search_applies_the_stage_lift_floor(monkeypatch):
    cases = [_period_case(0), _period_case(1)]
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        max_multi_start=1,
        bb_minimiser="rbf_surrogate",
    )

    def search(*, objective, args, **_kwargs):
        objective(np.array([0.5]), args)
        return ()

    monkeypatch.setattr(hp_execution, "run_hpr_candidate_search", search)
    floors_seen = []

    def period_objective(x, args, debug=False):
        floors_seen.append(np.asarray(args.stage_lift_floor).tolist())
        return HPRBackendResult.failure(reason="stop")

    with pytest.raises(HPRTargetingError):
        hp_execution.solve_hpr_multiperiod_placement(
            f_obj=period_objective,
            x0_ls=None,
            bnds=[(0.0, 1.0)],
            args=mp_args,
            stage_lifts=lambda x, args: np.array([_DECODED_LIFTS[args.period_idx]]),
        )

    assert floors_seen[0] == [10.0]
