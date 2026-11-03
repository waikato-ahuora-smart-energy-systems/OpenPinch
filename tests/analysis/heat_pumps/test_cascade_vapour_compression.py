from types import SimpleNamespace

import CoolProp.CoolProp as coolprop
import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.common._shared.plotting as hp_plotting
import OpenPinch.analysis.heat_pumps.common.shared as hp_shared
import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
import OpenPinch.analysis.targeting.cascade as target_cascade
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.common._shared.streams import (
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
)
from OpenPinch.analysis.heat_pumps.common.encoding import (
    decode_available_fractions,
    limit_available_duty,
)
from OpenPinch.contracts.hpr import HPRParsedState, HPRTopologyIdentifier

from .helpers import (
    _base_args,
    _patch_output_model_validate,
    _pt_with_hnet,
    _sc,
    _stream,
)


@pytest.mark.parametrize(
    ("n_cond", "n_evap", "expected"),
    [
        (1, 1, HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION),
        (2, 1, HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION),
        (1, 2, HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION),
    ],
)
def test_cascade_topology_identifies_scalar_cycle(n_cond, n_evap, expected):
    args = _base_args(n_cond=n_cond, n_evap=n_evap)

    assert hp_cascade._cascade_topology_id(args) is expected


def test_cascade_hp_x0_and_bounds_shapes_are_consistent():
    pytest.importorskip("CoolProp")

    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        H_cold=np.array([900.0, 450.0, 0.0]),
        H_hot=np.array([0.0, -500.0, -1000.0]),
        z_amb_cold=np.zeros(3),
        z_amb_hot=np.zeros(3),
        n_cond=2,
        n_evap=2,
        dt_range_max=130.0,
        dt_phase_change=1.0,
        T_env=20.0,
        Q_hpr_target=1000.0,
        refrigerant_ls=["R134A", "R134A", "R134A"],
        Q_cool_max=1000.0,
        Q_heat_max=900.0,
    )
    init = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        T_evap=np.array([50.0, 20.0]),
        Q_cond=np.array([600.0, 400.0]),
        Q_evap=np.array([500.0, 400.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    x0, bnds = hp_cascade._get_cascade_hp_opt_setup(init_res=init, args=args)

    assert x0.shape == (12,)
    assert len(bnds) == x0.shape[0]
    assert np.all(
        (x0[0] >= np.array([b[0] for b in bnds]))
        & (x0[0] <= np.array([b[1] for b in bnds]))
    )


def test_cascade_x0_round_trips_with_ambient_cooling_seed():
    args = _base_args(n_cond=2, n_evap=2)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=20.0,
    )

    x0, _ = hp_cascade._get_cascade_hp_opt_setup(init_res=init_res, args=args)
    vars = hp_cascade._parse_cascade_hp_state_variables(x0, args)

    np.testing.assert_allclose(vars.T_evap, init_res.T_evap)
    np.testing.assert_allclose(
        decode_available_fractions(vars.x_heat_split, vars.Q_heat_available),
        np.minimum(init_res.Q_cond, vars.Q_heat_available),
    )
    np.testing.assert_allclose(
        decode_available_fractions(vars.x_cool_split, vars.Q_cool_available),
        np.minimum(init_res.Q_evap[:-1], vars.Q_cool_available),
    )


def test_cascade_refrigeration_uses_full_evaporator_duty_seed():
    args = _base_args(n_cond=2, n_evap=2, is_heat_pumping=False)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=20.0,
        Q_amb_cold=0.0,
    )

    x0, _ = hp_cascade._get_cascade_hp_opt_setup(init_res=init_res, args=args)
    vars = hp_cascade._parse_cascade_hp_state_variables(x0, args)

    np.testing.assert_allclose(
        decode_available_fractions(vars.x_heat_split, vars.Q_heat_available),
        np.minimum(init_res.Q_cond[1:], vars.Q_heat_available),
    )
    np.testing.assert_allclose(
        decode_available_fractions(vars.x_cool_split, vars.Q_cool_available),
        np.minimum(init_res.Q_evap, vars.Q_cool_available),
    )
    # Heat controls are the condensers below the top stage, which takes the rest.
    np.testing.assert_allclose(
        vars.Q_heat_available,
        limit_available_duty(
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                vars.T_cond[1:], args.T_cold, args.H_cold, is_cond=True
            ),
            args.Q_heat_max,
        ),
    )


def test_cascade_x0_bounds_and_parse(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    x0, bnds = hp_cascade._get_cascade_hp_opt_setup(init_res=init_res, args=args)
    assert x0.shape[0] == 12

    monkeypatch.setattr(
        coolprop,
        "PropsSI",
        lambda prop, *_args: 200.0 if prop == "Tmin" else 422.0,
    )
    # The condensing caps read the cached critical point, not PropsSI.
    monkeypatch.setattr(hp_shared, "critical_point", lambda _ref: (422.0, 0.0, 0.0))
    assert len(bnds) == 12

    vars = hp_cascade._parse_cascade_hp_state_variables(
        np.array(
            [
                0.2,
                0.4,
                0.3,
                0.5,
                0.01,
                0.02,
                0.8,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ]
        ),
        args,
    )
    np.testing.assert_allclose(vars.T_cond, np.array([110.0, 78.0]))
    np.testing.assert_allclose(vars.T_evap, np.array([105.0, 70.0]))
    np.testing.assert_allclose(vars.dT_subcool, np.array([0.4, 0.16]))
    assert vars.Q_heat_base == pytest.approx(
        decode_available_fractions(vars.x_heat_split, vars.Q_heat_available).sum()
    )
    np.testing.assert_allclose(vars.x_heat_split, np.array([0.8, 0.1]))
    assert vars.Q_cool_base == pytest.approx(
        decode_available_fractions(vars.x_cool_split, vars.Q_cool_available).sum()
    )
    np.testing.assert_allclose(vars.x_cool_split, np.array([0.1]))


def test_compute_cascade_hp_system_obj_unsolved_and_solved(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    x = np.array([0.2] * 9)
    monkeypatch.setattr(
        hp_cascade,
        "_parse_cascade_hp_state_variables",
        lambda _x, _args: HPRParsedState.model_validate(
            {
                "T_cond": np.array([110.0, 90.0]),
                "dT_subcool": np.array([5.0, 5.0]),
                "T_evap": np.array([70.0, 50.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
                "dT_ihx_gas_side": np.array([0.0, 0.0, 0.0]),
                "Q_heat_base": 200.0,
                "x_heat_split": np.array([0.6, 1.0]),
                "Q_heat_available": np.array([120.0, 80.0]),
                "Q_cool_base": 50.0,
                "x_cool_split": np.array([1.0]),
                "Q_cool_available": np.array([50.0]),
            }
        ),
    )

    class _FakeCascadeUnsolved:
        solved = False
        work = 50.0

        def solve(self, **kwargs):
            return None

    monkeypatch.setattr(
        hp_cascade, "CascadeVapourCompressionCycle", _FakeCascadeUnsolved
    )
    out_unsolved = hp_cascade._compute_cascade_hp_system_obj(x, args)
    assert np.isinf(out_unsolved.obj)

    class _FakeCascadeSolved:
        solved = True
        work = 40.0
        work_arr = np.array([10.0, 30.0])
        Q_cool = 70.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([50.0, 20.0])
        dT_subcool = np.array([5.0, 5.0])
        penalty = 1.0

        def solve(self, **kwargs):
            return None

        def build_stream_collection(self, **kwargs):
            return _sc(_stream("HP", 100.0, 90.0, 10.0, is_process_stream=False))

    monkeypatch.setattr(hp_cascade, "CascadeVapourCompressionCycle", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(5.0, 1.0), _pt_with_hnet(6.0, 2.0)])
    monkeypatch.setattr(
        target_cascade, "get_process_heat_cascade", lambda **kwargs: next(seq)
    )
    calls = {"plot": 0}
    monkeypatch.setattr(
        hp_plotting,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: calls.__setitem__("plot", calls["plot"] + 1),
    )

    out = hp_cascade._compute_cascade_hp_system_obj(x, args, debug=True)
    assert len(out.hpr_hot_streams) == 1
    assert out.Q_ext == pytest.approx(7.0)
    assert out.utility_tot == pytest.approx(47.0)
    assert out.w_net == pytest.approx(40.0)
    assert out.cop_h == pytest.approx(5.0)
    assert calls["plot"] == 1


def test_compute_cascade_refrigeration_obj_solves_refrigeration_mode(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1, is_heat_pumping=False)
    captured = {}
    monkeypatch.setattr(
        hp_cascade,
        "_parse_cascade_hp_state_variables",
        lambda _x, _args: HPRParsedState.model_validate(
            {
                "T_cond": np.array([100.0]),
                "dT_subcool": np.array([5.0]),
                "T_evap": np.array([60.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
                "dT_ihx_gas_side": np.array([0.0]),
                "Q_heat_base": None,
                "x_heat_split": None,
                "Q_heat_available": None,
                "Q_cool_base": 80.0,
                "x_cool_split": np.array([1.0]),
                "Q_cool_available": np.array([80.0]),
            }
        ),
    )

    class _FakeCascadeSolved:
        solved = True
        work = 20.0
        work_arr = np.array([20.0])
        Q_heat_arr = np.array([100.0])
        Q_cool_arr = np.array([80.0])
        dT_subcool = np.array([0.0])
        penalty = 0.0

        def solve(self, **kwargs):
            captured.update(kwargs)

        def build_stream_collection(self, **kwargs):
            return _sc(_stream("HP", 90.0, 80.0, 10.0, is_process_stream=False))

    monkeypatch.setattr(hp_cascade, "CascadeVapourCompressionCycle", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(1.0, 1.0), _pt_with_hnet(1.0, 1.0)])
    monkeypatch.setattr(
        target_cascade, "get_process_heat_cascade", lambda **kwargs: next(seq)
    )

    out = hp_cascade._compute_cascade_hp_system_obj(np.array([0.2]), args)

    assert captured["dtcont"] == pytest.approx(args.dtcont_hp)
    assert captured["is_heat_pump"] is False
    duty = captured["duty_allocation"]
    assert duty.heat.Q_base is None
    assert duty.heat.x_split is None
    assert duty.heat.Q_available is None
    assert duty.cool.Q_base == pytest.approx(80.0)
    np.testing.assert_allclose(duty.cool.x_split, np.array([1.0]))
    np.testing.assert_allclose(duty.cool.Q_available, np.array([80.0]))
    assert out.cop_h == pytest.approx(4.0)
    record = out.target_simulation_record
    assert record is not None
    assert record.simulation_backend == "coolprop"
    assert record.mode == "refrigeration"
    assert record.nominal_useful_duty == pytest.approx(80.0)
    assert record.refrigerant_spec == args.refrigerant_ls[0]


def test_cascade_x0_branch_for_single_stage():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        dt_phase_change=20.0,
        dt_range_max=10.0,
        T_cold=np.array([50.0, 45.0]),
        T_hot=np.array([40.0, 35.0]),
        H_cold=np.array([100.0, 0.0]),
        H_hot=np.array([0.0, -100.0]),
        z_amb_cold=np.zeros(2),
        z_amb_hot=np.zeros(2),
        Q_hpr_target=100.0,
        T_env=20.0,
        refrigerant_ls=["R134A", "R134A"],
        Q_cool_max=100.0,
        Q_heat_max=100.0,
    )
    init_res = SimpleNamespace(
        T_cond=np.array([46.0]),
        Q_cond=np.array([70.0]),
        T_evap=np.array([36.0]),
        Q_evap=np.array([65.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    x0, _ = hp_cascade._get_cascade_hp_opt_setup(init_res=init_res, args=args)
    assert x0.shape == (5,)


def test_cascade_optimiser_allows_missing_initial_seed(monkeypatch):
    captured = {}
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_cycle=False)
    _patch_output_model_validate(monkeypatch)
    monkeypatch.setattr(
        hp_cascade,
        "preflight_coolprop_hpr_targeting",
        lambda **_kwargs: None,
    )

    monkeypatch.setattr(
        hp_cascade,
        "validate_vapour_hp_refrigerant_ls",
        lambda num_stages, args: ["R134A"] * num_stages,
    )

    def fake_solve_hpr_placement(*, f_obj, x0_ls, bnds, args):
        captured["x0_ls"] = x0_ls
        return SimpleNamespace(success=True)

    monkeypatch.setattr(hp_cascade, "solve_hpr_placement", fake_solve_hpr_placement)

    out = hp_cascade.optimise_cascade_heat_pump_placement(args)

    assert captured["x0_ls"] is None
    assert out.success is True


def test_cascade_subcooling_scales_with_the_lift_to_the_coldest_evaporator():
    args = _base_args(n_cond=1, n_evap=1)
    args.refrigerant_ls = ["water"]
    _, bnds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    x = np.zeros(len(bnds))
    layout = hp_cascade.HPRoptVectorLayout(
        n_cond=1, n_evap=1, n_subcool=1, n_heat_split=1, n_cool_split=0, n_ihx=1
    )
    x[layout.subcool_slice] = 0.5

    state = hp_cascade._parse_cascade_hp_state_variables(x, args)

    # The same limit as a parallel machine: the liquid can cool to the evaporator.
    assert state.dT_subcool[0] == pytest.approx(
        0.5 * (state.T_cond[0] - state.T_evap[0])
    )


def test_cascade_parse_floors_evaporation_at_the_refrigerant_minimum():
    args = _base_args(n_cond=1, n_evap=1)
    args.T_hot = np.array([30.0, 10.0, -20.0])
    args.refrigerant_ls = ["water"]
    _, bnds = hp_cascade._get_cascade_hp_opt_setup(None, args)

    # x_evap = 0 decodes to the coldest source temperature, -20 degC.
    state = hp_cascade._parse_cascade_hp_state_variables(np.zeros(len(bnds)), args)

    assert state.T_evap[0] == pytest.approx(0.01, abs=1e-5)


def test_cascade_process_evaporator_uses_the_bottom_stage_refrigerant_minimum():
    args = _base_args(n_cond=2, n_evap=1)
    args.T_hot = np.array([30.0, 10.0, -20.0])
    args.refrigerant_ls = ["water", "ammonia"]
    _, bnds = hp_cascade._get_cascade_hp_opt_setup(None, args)

    state = hp_cascade._parse_cascade_hp_state_variables(np.zeros(len(bnds)), args)

    # The process evaporator belongs to the ammonia stage, not the water stage.
    assert state.T_evap[0] == pytest.approx(-20.0)


def test_cascade_state_reports_the_subcooling_the_cycle_applied():
    pytest.importorskip("CoolProp")
    state = HPRParsedState.model_validate(
        {
            "T_cond": np.array([84.9, 84.8]),
            "dT_subcool": np.array([0.0, 85.0]),
            "T_evap": np.array([-0.2]),
            "dT_ihx_gas_side": np.array([0.0, 0.0]),
            "Q_heat_base": 250.0,
            "x_heat_split": np.array([1.0, 0.0]),
            "Q_heat_available": np.array([250.0, 250.0]),
        }
    )
    args = SimpleNamespace(
        dtcont_hp=5.0,
        eta_comp=0.7,
        refrigerant_ls=["Ammonia", "R1234ze(E)"],
        dt_cascade_hx=5.0,
    )

    cycle = hp_cascade._build_and_solve_cascade_cycle(state, args, True)

    assert cycle.solved
    assert state.dT_subcool == pytest.approx([0.0, 0.0], abs=1e-6)


def test_cascade_refrigeration_heat_controls_sit_below_the_top_stage():
    pytest.importorskip("CoolProp")
    args = SimpleNamespace(
        dtcont_hp=5.0,
        eta_comp=0.7,
        refrigerant_ls=["Ammonia", "Ammonia"],
        dt_cascade_hx=5.0,
    )
    state = HPRParsedState.model_validate(
        {
            "T_cond": np.array([40.0, 10.0]),
            "dT_subcool": np.array([0.0, 0.0]),
            "T_evap": np.array([-20.0]),
            "dT_ihx_gas_side": np.array([0.0, 0.0]),
            "Q_heat_base": 50.0,
            "x_heat_split": np.array([1.0]),
            "Q_heat_available": np.array([50.0]),
            "Q_cool_base": 100.0,
            "x_cool_split": np.array([1.0]),
            "Q_cool_available": np.array([100.0]),
        }
    )

    cycle = hp_cascade._build_and_solve_cascade_cycle(state, args, False)

    assert cycle.solved
    warm, cold = cycle.subcycles
    assert cold.T_cond == pytest.approx(10.0)
    assert cold.Q_heat == pytest.approx(50.0)
    assert cold.Q_cool == pytest.approx(100.0)
    assert warm.Q_cool == pytest.approx(0.0)
    assert warm.Q_heat == pytest.approx(warm.Q_cond)
    assert cycle.Q_heat == pytest.approx(cycle.Q_cool + cycle.work)


def test_cascade_one_condenser_delivers_its_duty_once_with_two_evaporators():
    pytest.importorskip("CoolProp")
    args = SimpleNamespace(
        dtcont_hp=5.0,
        eta_comp=0.7,
        refrigerant_ls=["Ammonia", "Ammonia"],
        dt_cascade_hx=1.0,
    )
    state = HPRParsedState.model_validate(
        {
            "T_cond": np.array([80.0]),
            "dT_subcool": np.array([0.0]),
            "T_evap": np.array([24.0, 5.0]),
            "dT_ihx_gas_side": np.array([0.0, 0.0]),
            "Q_heat_base": 100.0,
            "x_heat_split": np.array([1.0]),
            "Q_heat_available": np.array([100.0]),
            "Q_cool_base": 50.0,
            "x_cool_split": np.array([1.0]),
            "Q_cool_available": np.array([50.0]),
        }
    )

    cycle = hp_cascade._build_and_solve_cascade_cycle(state, args, True)

    assert cycle.solved
    # The lower stage condenses at 25 degC into the upper stage's evaporator.
    assert cycle.Q_heat_arr == pytest.approx([100.0, 0.0])
    # The upper evaporator takes 50 from the process; the lower stage covers
    # the rest of its evaporator through the cascade and balances the cooling.
    upper, lower = cycle.subcycles
    assert upper.Q_cool == pytest.approx(50.0)
    assert lower.Q_cond == pytest.approx(upper.Q_cas_cool)
    assert lower.Q_cool == pytest.approx(lower.Q_evap)
    assert cycle.Q_heat == pytest.approx(cycle.Q_cool + cycle.work)


@pytest.mark.parametrize("evaporators", [2, 3])
def test_public_target_with_one_condenser_has_no_phantom_process_condensers(
    evaporators,
):
    pytest.importorskip("CoolProp")
    target = PinchProblem(
        "OpenPinch/tutorials/sample_cases/chocolate_factory.json", project_name="m"
    ).target.vapour_compression_heat_pump(
        load_fraction=0.25,
        condensers=1,
        evaporators=evaporators,
        refrigerants=["Ammonia"],
        maximum_restarts=1,
        maximum_iterations=20,
        maximum_evaluations=50,
    )

    details = target.hpr_details
    Q_heat = np.asarray(details.Q_heat, dtype=float)
    assert Q_heat.size == evaporators
    assert Q_heat[1:] == pytest.approx(np.zeros(evaporators - 1))
    assert target.hpr_cop == pytest.approx(Q_heat[0] / target.hpr_work, rel=1e-9)
