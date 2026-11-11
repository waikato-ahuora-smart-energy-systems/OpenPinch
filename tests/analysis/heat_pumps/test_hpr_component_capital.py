"""Component capital: compressors, motors and plate exchangers (Ommen 2015).

Purchased equipment cost follows PEC = PEC_ref (X / X_ref)^a (Ommen et al.,
Int. J. Refrig. 2015, Table 4, 2013 EUR). Installed capital is 4.16 x PEC
(Bejan et al. 1996), scaled to the IEA HPT Annex 68 cost level.
"""

from __future__ import annotations

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
from OpenPinch.analysis.heat_pumps.common.capital import (
    ANNEX_68_COST_AT_1_MW,
    REFERENCE_MACHINE,
    compressor_pec,
    exchanger_pec,
    installed_capital,
    matched_area,
    motor_pec,
)
from OpenPinch.analysis.heat_pumps.common.machine_capital import (
    machine_components,
    process_side_areas,
)
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection

from .helpers import _base_args


@pytest.mark.parametrize(
    ("fluid", "volume_flow", "pec"),
    [
        ("Ammonia", 178.4, 11914.0),
        ("Propane", 279.8, 19850.0),
        ("IsoButane", 279.8, 19850.0),
        ("R134a", 178.4, 10631.0),
        ("CO2", 25.6, 12109.0),
        # Other fluids are priced as the R134a family.
        ("R1234ze(E)", 178.4, 10631.0),
    ],
)
def test_a_compressor_is_priced_on_its_suction_volume_flow(fluid, volume_flow, pec):
    assert compressor_pec(fluid, volume_flow) == pytest.approx(pec)


def test_a_suction_volume_beyond_one_compressor_takes_several_in_parallel():
    # 280 m3/h is the largest screw compressor priced; 560 m3/h takes two.
    assert compressor_pec("Propane", 560.0) == pytest.approx(
        2 * 19850.0 * (280.0 / 279.8) ** 0.73
    )


def test_motors_are_priced_on_shaft_power_except_for_hydrocarbon_compressors():
    assert motor_pec("Ammonia", 250.0) == pytest.approx(10710.0)
    assert motor_pec("Propane", 250.0) == 0.0


def test_water_compressors_are_priced_as_the_zuhlsdorf_turbo_compressors():
    # Zuhlsdorf et al. (2018): two R718 turbo compressors of 1231 and 468 m3/h
    # cost 50,000 EUR with their motors, gears and inverters.
    assert compressor_pec("Water", 1231.0) + compressor_pec(
        "Water", 468.0
    ) == pytest.approx(50_000.0)
    assert motor_pec("Water", 250.0) == 0.0


def test_a_water_turbo_compressor_takes_a_large_volume_flow_in_one_unit():
    assert compressor_pec("Water", 200_000.0) == pytest.approx(
        compressor_pec("Water", 100_000.0) * 2.0**0.73
    )


def test_a_plate_exchanger_is_priced_on_its_area():
    assert exchanger_pec(42.0) == pytest.approx(15526.0)
    assert exchanger_pec(84.0) == pytest.approx(15526.0 * 2.0**0.8)


def test_the_reference_machine_costs_the_annex_68_level():
    # Ommen's worked example: 1 MW ammonia heat pump, 20 K lift.
    assert installed_capital(REFERENCE_MACHINE) == pytest.approx(ANNEX_68_COST_AT_1_MW)


def _ammonia_stage():
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=5.0,
        T_cond=55.0,
        dtcont=0.0,
        eta_motor=0.95,
        refrigerant="Ammonia",
        dT_ihx_gas_side=10.0,
        Q_heat=1000.0,
    )
    assert cycle.solved
    return cycle


def test_a_stage_reports_its_suction_volume_flow_and_shaft_power():
    cycle = _ammonia_stage()

    P_low = CP.PropsSI("P", "T", 278.15, "Q", 1, "Ammonia")
    h_suction = CP.PropsSI("H", "P", P_low, "T", 288.15, "Ammonia")
    rho_suction = CP.PropsSI("D", "P", P_low, "T", 288.15, "Ammonia")
    h_out = cycle.Hs[2]
    h_discharge = cycle.Hs[1]
    mass_flow = 1000.0e3 / (h_discharge - h_out)  # kg/s
    assert cycle.suction_volume_flow == pytest.approx(
        mass_flow / rho_suction * 3600.0, rel=1e-6
    )
    assert cycle.shaft_power == pytest.approx(cycle.work * 0.95)
    assert cycle.ihx_duty == pytest.approx(
        mass_flow * (h_suction - CP.PropsSI("H", "P", P_low, "Q", 1, "Ammonia")) / 1e3,
        rel=1e-6,
    )


def test_a_stage_reports_its_ihx_log_mean_temperature_difference():
    cycle = _ammonia_stage()

    # Liquid 55 C -> T_out against gas 5 -> 15 C, counter-current.
    T_liquid_out = cycle.Ts[3 - 1] - 273.15  # condenser outlet = IHX liquid inlet
    hot_end = T_liquid_out - 15.0
    cold_end = cycle.ihx_liquid_outlet_temperature - 5.0
    assert cycle.ihx_lmtd == pytest.approx(
        (hot_end - cold_end) / np.log(hot_end / cold_end), rel=1e-6
    )


def _uniform(u):
    return lambda t_h1, t_h2, t_c1, t_c2: np.full(np.shape(t_h1), u)


def test_a_condenser_area_matches_its_curve_against_the_sink_interval_by_interval():
    # 100 kW condensing at 80 C into a sink heated 50 -> 70 C: ends 10 K and
    # 30 K, log-mean 20 / ln 3.
    area = matched_area(
        T_hot=np.array([80.0, 80.0]),
        H_hot=np.array([100.0, 0.0]),
        T_cold=np.array([70.0, 50.0]),
        H_cold=np.array([100.0, 0.0]),
        extra_dt=0.0,
        heat_transfer_coefficient=_uniform(2000.0),
    )

    assert area == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(3.0)))


def test_the_process_streams_approach_widens_every_interval():
    area = matched_area(
        T_hot=np.array([80.0, 80.0]),
        H_hot=np.array([100.0, 0.0]),
        T_cold=np.array([70.0, 50.0]),
        H_cold=np.array([100.0, 0.0]),
        extra_dt=2.0,
        heat_transfer_coefficient=_uniform(2000.0),
    )

    assert area == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(32.0 / 12.0)))


def test_touching_curves_need_an_infinite_area():
    area = matched_area(
        T_hot=np.array([80.0, 80.0]),
        H_hot=np.array([100.0, 0.0]),
        T_cold=np.array([80.0, 50.0]),
        H_cold=np.array([100.0, 0.0]),
        extra_dt=0.0,
        heat_transfer_coefficient=_uniform(2000.0),
    )

    assert area == float("inf")


def _condenser_stream(T_in, T_out, duty):
    streams = StreamCollection()
    streams.add(
        Stream(
            name="Condenser",
            supply_temperature=T_in,
            target_temperature=T_out,
            heat_flow=duty,
            delta_t_contribution=0.0,
            heat_transfer_coefficient=1.0,
            is_process_stream=False,
        )
    )
    return streams


def test_the_condenser_area_is_matched_against_the_top_of_the_sink():
    args = _base_args(
        T_cold=np.array([70.0, 50.0]),
        H_cold=np.array([100.0, 0.0]),
        dtcont_hp=0.0,
        process_dt_cont=0.0,
        u_two_phase=2000.0,
        u_liquid=1000.0,
        u_gas=400.0,
    )

    condenser, evaporator, *_ = process_side_areas(
        _condenser_stream(80.0, 79.99, 100.0), args, condensing_temperatures=[80.0]
    )

    assert condenser == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(3.0)), rel=1e-3)
    assert evaporator == 0.0


def test_a_cascade_machine_has_both_compressors_and_its_cascade_exchanger():
    cascade = CascadeVapourCompressionCycle()
    cascade.solve(
        T_evap=np.array([5.0]),
        T_cond=np.array([80.0, 45.0]),
        dtcont=0.0,
        refrigerant=["Ammonia", "Ammonia"],
        dT_ihx_gas_side=0.0,
        dt_cascade_hx=5.0,
        Q_heat=np.array([100.0, 0.0]),
    )
    args = _base_args(
        dt_cascade_hx=5.0, u_two_phase=2000.0, u_liquid=1000.0, u_gas=400.0
    )

    (machine,) = machine_components(
        [cascade._subcycles], process_areas=(10.0, 20.0, 0.0, 0.0), args=args
    )

    assert [c.fluid for c in machine.compressors] == ["Ammonia", "Ammonia"]
    assert [c.volume_flow for c in machine.compressors] == pytest.approx(
        [stage.suction_volume_flow for stage in cascade._subcycles]
    )
    cascade_heat = cascade._subcycles[1].Q_cas_heat
    assert cascade_heat > 0.0
    assert sorted(machine.exchanger_areas) == pytest.approx(
        sorted([10.0, 20.0, cascade_heat * 1e3 / (2000.0 * 5.0)])
    )


def test_a_heat_pumps_capital_is_its_components_installed_capital():
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]
    state = hp_cascade._parse_cascade_hp_state_variables(
        np.full(len(hp_cascade._get_cascade_hp_opt_setup(None, args)[1]), 0.4), args
    )
    cycle = hp_cascade._build_and_solve_cascade_cycle(state, args, True)
    assert cycle.solved
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)

    result = hp_cascade._compute_cascade_hp_system_obj(
        np.full(len(hp_cascade._get_cascade_hp_opt_setup(None, args)[1]), 0.4), args
    )

    areas = process_side_areas(
        streams, args, condensing_temperatures=np.asarray(state.T_cond)
    )
    (machine,) = machine_components([cycle._subcycles], process_areas=areas, args=args)
    assert result.hpr_machine_capital_costs == pytest.approx(
        (installed_capital(machine),)
    )


def test_the_evaluator_sizes_the_air_exchangers_for_the_cascades_air_duties(
    monkeypatch,
):
    from OpenPinch.analysis.heat_pumps.common import shared

    cascades, calls = [], []
    cascade_air_duties = shared._cascade_air_duties
    process_side_areas_ = shared.process_side_areas

    def record_cascade(*a, **k):
        cascades.append(cascade_air_duties(*a, **k))
        return cascades[-1]

    def record_areas(*a, **k):
        calls.append((k["air_duties"], k["wrong_side_duties"]))
        return process_side_areas_(*a, **k)

    monkeypatch.setattr(shared, "_cascade_air_duties", record_cascade)
    monkeypatch.setattr(shared, "process_side_areas", record_areas)
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]

    hp_cascade._compute_cascade_hp_system_obj(
        np.full(len(hp_cascade._get_cascade_hp_opt_setup(None, args)[1]), 0.4), args
    )

    condenser_cascade, evaporator_cascade = cascades[-2:]
    assert calls[-1] == (
        (condenser_cascade.Q_air_sink, evaporator_cascade.Q_air_source),
        (condenser_cascade.Q_ext_bottom, evaporator_cascade.Q_ext_top),
    )


def _target_inputs(options):
    from OpenPinch.analysis.heat_pumps.common import preprocessing as hp_pre
    from OpenPinch.domain.configuration import Configuration

    return hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        config=Configuration(options={"HPR_N_COND": 1, "HPR_N_EVAP": 1, **options}),
    )


def test_the_heat_transfer_coefficients_and_process_approach_come_from_settings():
    args = _target_inputs(
        {
            "HPR_U_TWO_PHASE": 2500.0,
            "HPR_U_LIQUID": 900.0,
            "HPR_U_GAS": 350.0,
            "THERMAL_DT_CONT": 6.0,
        }
    )

    assert (args.u_two_phase, args.u_liquid, args.u_gas) == (2500.0, 900.0, 350.0)
    assert args.process_dt_cont == 6.0
    defaults = _target_inputs({})
    assert (defaults.u_two_phase, defaults.u_liquid, defaults.u_gas) == (
        2000.0,
        1000.0,
        400.0,
    )


def test_a_condenser_below_the_top_of_the_sink_heats_the_sink_just_below_it():
    # The sink takes 100 kW from 50 to 90 C. A condenser at 70 C cannot heat
    # its top; it heats the closest 25 kW it can, 60 -> 70 C.
    args = _base_args(
        T_cold=np.array([90.0, 50.0]),
        H_cold=np.array([100.0, 0.0]),
        dtcont_hp=0.0,
        process_dt_cont=5.0,
        u_two_phase=2000.0,
        u_liquid=1000.0,
        u_gas=400.0,
    )

    condenser, *_ = process_side_areas(
        _condenser_stream(70.0, 69.99, 25.0), args, condensing_temperatures=[70.0]
    )

    # Ends: 70 - 70 + 5 = 5 K and 70 - 60 + 5 = 15 K.
    assert condenser == pytest.approx(25e3 / (2000.0 * 10.0 / np.log(3.0)), rel=1e-3)


def _evaporator_stream(T_in, T_out, duty):
    streams = StreamCollection()
    streams.add(
        Stream(
            name="Evaporator",
            supply_temperature=T_in,
            target_temperature=T_out,
            heat_flow=duty,
            delta_t_contribution=0.0,
            heat_transfer_coefficient=1.0,
            is_process_stream=False,
        )
    )
    return streams


def _air_args(**overrides):
    # Air takes heat at or above 30 C and supplies it at or below 10 C (shifted).
    return _base_args(
        **{
            "dtcont_hp": 0.0,
            "process_dt_cont": 0.0,
            "T_env": 20.0,
            "dt_env_cont": 10.0,
            "u_two_phase": 2000.0,
            "u_liquid": 1000.0,
            "u_gas": 400.0,
            "u_air": 50.0,
            **overrides,
        }
    )


def test_condenser_heat_rejected_to_air_is_sized_against_the_air():
    # The sink takes 100 kW from 50 to 70 C; the other 50 kW goes to air.
    args = _air_args(T_cold=np.array([70.0, 50.0]), H_cold=np.array([100.0, 0.0]))

    condenser, evaporator, air_cooler, air_heater = process_side_areas(
        _condenser_stream(80.0, 79.99, 150.0),
        args,
        condensing_temperatures=[80.0],
        air_duties=(50.0, 0.0),
    )

    assert condenser == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(3.0)), rel=1e-3)
    # 80 C against air at 30 C, with the air approach (10 K) added back. The
    # air side's coefficient is in series with the condensing side's.
    u = 1.0 / (1.0 / 2000.0 + 1.0 / 50.0)
    assert air_cooler == pytest.approx(50e3 / (u * 60.0), rel=1e-3)
    assert (evaporator, air_heater) == (0.0, 0.0)


def test_evaporator_heat_drawn_from_air_is_sized_against_the_air():
    # The source gives 100 kW from 30 to 10 C; the other 50 kW comes from air.
    args = _air_args(T_hot=np.array([30.0, 10.0]), H_hot=np.array([100.0, 0.0]))

    condenser, evaporator, air_cooler, air_heater = process_side_areas(
        _evaporator_stream(-0.01, 0.0, 150.0),
        args,
        condensing_temperatures=[80.0],
        air_duties=(0.0, 50.0),
    )

    assert evaporator == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(3.0)), rel=1e-3)
    u = 1.0 / (1.0 / 2000.0 + 1.0 / 50.0)
    assert air_heater == pytest.approx(50e3 / (u * 20.0), rel=1e-3)
    assert (condenser, air_cooler) == (0.0, 0.0)


def test_heat_on_the_wrong_side_of_the_process_has_no_process_exchanger():
    # 20 kW of condensing heat lands below the sink, on the wrong side, and is
    # penalised; only the other 100 kW heats the sink.
    args = _air_args(T_cold=np.array([70.0, 50.0]), H_cold=np.array([100.0, 0.0]))

    condenser, *_ = process_side_areas(
        _condenser_stream(80.0, 79.99, 120.0),
        args,
        condensing_temperatures=[80.0],
        wrong_side_duties=(20.0, 0.0),
    )

    assert condenser == pytest.approx(100e3 / (2000.0 * 20.0 / np.log(3.0)), rel=1e-3)


def test_a_round_off_sliver_of_process_duty_needs_no_exchanger():
    # The cascade places all but a round-off sliver of the condensing heat in
    # air; the sliver, too small to place on the sink, is not an exchanger.
    args = _air_args(T_cold=np.array([90.0, 85.0]), H_cold=np.array([100.0, 0.0]))

    condenser, _, air_cooler, _ = process_side_areas(
        _condenser_stream(80.0, 79.99, 1000.0),
        args,
        condensing_temperatures=[80.0],
        air_duties=(999.99, 0.0),
    )

    assert condenser == 0.0
    assert air_cooler > 0.0


def test_the_air_heat_transfer_coefficient_comes_from_settings():
    assert _target_inputs({"HPR_U_AIR": 40.0}).u_air == 40.0
    assert _target_inputs({}).u_air == 50.0


@pytest.mark.parametrize(
    "key",
    [
        "COSTING_HPR_EQUIPMENT_COST",
        "COSTING_HPR_INSTALLATION_FACTOR",
        "COSTING_HPR_COST_EXP",
        "COSTING_HPR_COST_FIXED_SHARE",
        "COSTING_HPR_COST_STAGE_SHARE",
        "COSTING_HPR_COST_TEMP_FACTOR",
        "COSTING_HPR_COST_TEMP_BASE",
    ],
)
def test_the_duty_based_capital_settings_are_gone(key):
    from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS

    assert key not in CONFIG_FIELD_SPECS
