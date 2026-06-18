from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import HPRcycle
from OpenPinch.lib.schemas.hpr import HPRBackendResult, HPRThermoArtifacts
from OpenPinch.services.heat_pump_integration import (
    heat_pump_and_refrigeration_entry as hp,
)
from OpenPinch.services.heat_pump_integration.common import shared as hp_shared
from OpenPinch.services.heat_pump_integration.common.encoding import decode_duty_splits
from OpenPinch.services.heat_pump_integration.common.preprocessing import (
    construct_HPRTargetInputs,
)
from OpenPinch.services.heat_pump_integration.targeting_services import (
    vapour_compression_mvr as hp_vc_mvr,
)
from OpenPinch.services.heat_pump_integration.unit_models import (
    vapour_compression_mvr_cascade as vcmvr_cascade_mod,
)
from OpenPinch.services.heat_pump_integration.unit_models.mechanical_vapour_recompression_cycle import (
    MechanicalVapourRecompressionCycle,
)
from OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_mvr_cascade import (
    VapourCompressionMvrCascade,
)

from ..helpers import (
    _base_args,
    _patch_output_model_validate,
    _pt_with_hnet,
    _sc,
    _stream,
)


def test_mvr_water_default_solves_and_balances_energy():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()

    assert not callable(mvr.solve)

    work = mvr.solve_from_source_heat(
        T_evap=80.0,
        T_cond=95.0,
        Q_source=1000.0,
    )

    assert mvr.solved is True
    assert work > 0.0
    assert mvr.Q_evap == pytest.approx(1000.0)
    assert mvr.Q_cond > 0.0
    assert mvr.work == pytest.approx(work)
    assert mvr.COP_h == pytest.approx(mvr.Q_cond / mvr.work)
    assert mvr.cycle_states[0]["T"] == pytest.approx(80.0 + 273.15)


def test_mvr_accepts_generic_coolprop_fluid_and_motor_efficiency_changes_work():
    pytest.importorskip("CoolProp")
    ideal_motor = MechanicalVapourRecompressionCycle()
    lossy_motor = MechanicalVapourRecompressionCycle()

    ideal_motor.solve_from_mass_flow(
        T_evap=-20.0,
        T_cond=10.0,
        m_dot=0.01,
        eta_mvr_comp=0.75,
        eta_motor=1.0,
        fluid="R134A",
    )
    lossy_motor.solve_from_mass_flow(
        T_evap=-20.0,
        T_cond=10.0,
        m_dot=0.01,
        eta_mvr_comp=0.75,
        eta_motor=0.5,
        fluid="R134A",
    )

    assert ideal_motor.solved is True
    assert lossy_motor.solved is True
    assert lossy_motor.shaft_work == pytest.approx(ideal_motor.shaft_work)
    assert lossy_motor.work == pytest.approx(ideal_motor.work / 0.5)


def test_mvr_supports_superheated_inlet_and_subcooled_liquid_outlet():
    pytest.importorskip("CoolProp")
    base = MechanicalVapourRecompressionCycle()
    modified = MechanicalVapourRecompressionCycle()

    base.solve_from_mass_flow(T_evap=80.0, T_cond=95.0, m_dot=0.01)
    modified.solve_from_mass_flow(
        T_evap=80.0,
        T_cond=95.0,
        m_dot=0.01,
        dT_superheat=8.0,
        dT_subcool=6.0,
    )

    assert modified.solved is True
    assert modified.dT_superheat == pytest.approx(8.0)
    assert modified.dT_subcool == pytest.approx(6.0)
    assert modified.cycle_states[0]["T"] == pytest.approx(80.0 + 273.15 + 8.0)
    assert modified.cycle_states[2]["T"] == pytest.approx(95.0 + 273.15 - 6.0)
    assert modified.Q_cond != pytest.approx(base.Q_cond)


def test_mvr_open_stage_source_heat_and_heat_components_balance():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()

    mvr.solve_from_source_heat(
        T_evap=90.0,
        T_cond=105.0,
        Q_source=1000.0,
        dT_subcool=4.0,
        eta_mvr_comp=0.75,
        eta_motor=0.95,
        process_split=0.35,
    )

    assert mvr.solved is True
    assert mvr.Q_evap == pytest.approx(1000.0)
    assert mvr.q_source > 0.0
    assert mvr.q_desuperheat == pytest.approx(0.0)
    assert mvr.q_liquid_injection > 0.0
    assert mvr.liquid_injection_ratio > 0.0
    assert mvr.m_dot > mvr.source_m_dot
    assert mvr.q_latent_condense > 0.0
    assert mvr.q_subcool_process > 0.0
    assert mvr.Q_cond == pytest.approx(
        mvr.source_m_dot
        * (mvr.q_desuperheat + mvr.q_latent_condense + mvr.q_subcool_process)
    )
    components = mvr.process_heat_components()
    assert mvr.process_split == pytest.approx(0.35)
    assert components["desuperheat"] == pytest.approx(0.0)
    assert components["latent"] == pytest.approx(
        mvr.source_m_dot * 0.35 * mvr.q_latent_condense
    )
    assert components["subcool"] == pytest.approx(
        mvr.source_m_dot * 0.35 * mvr.q_subcool_process
    )
    assert components["total"] == pytest.approx(
        components["desuperheat"] + components["latent"] + components["subcool"]
    )
    assert mvr.process_heat == pytest.approx(components["total"])
    assert mvr.COP_process_h == pytest.approx(mvr.process_heat / mvr.work)
    assert mvr.process_m_dot_out == pytest.approx(mvr.m_dot * 0.65)
    assert mvr.shaft_work == pytest.approx(mvr.work * mvr.eta_motor)


def test_mvr_source_heat_supports_superheated_inlet():
    pytest.importorskip("CoolProp")
    base = MechanicalVapourRecompressionCycle()
    superheated = MechanicalVapourRecompressionCycle()

    base.solve_from_source_heat(T_evap=85.0, T_cond=100.0, Q_source=1000.0)
    superheated.solve_from_source_heat(
        T_evap=85.0,
        T_cond=100.0,
        Q_source=1000.0,
        dT_superheat=6.0,
    )

    assert superheated.solved is True
    assert superheated.dT_superheat == pytest.approx(6.0)
    assert superheated.cycle_states[0]["T"] == pytest.approx(85.0 + 273.15 + 6.0)
    assert superheated.Q_evap == pytest.approx(1000.0)
    assert superheated.source_m_dot != pytest.approx(base.source_m_dot)
    assert superheated.work != pytest.approx(base.work)


def test_mvr_liquid_injection_changes_vapour_mass_and_external_desuperheat():
    pytest.importorskip("CoolProp")
    dry = MechanicalVapourRecompressionCycle()
    injected = MechanicalVapourRecompressionCycle()

    dry.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        eta_mvr_comp=0.75,
        liquid_injection=False,
    )
    injected.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        eta_mvr_comp=0.75,
        liquid_injection=True,
    )

    assert dry.solved is True
    assert injected.solved is True
    assert dry.source_m_dot == pytest.approx(dry.m_dot)
    assert dry.liquid_injection_ratio == pytest.approx(0.0)
    assert dry.q_desuperheat > 0.0
    assert dry.q_liquid_injection == pytest.approx(0.0)
    assert injected.source_m_dot == pytest.approx(dry.source_m_dot)
    assert injected.m_dot > injected.source_m_dot
    assert injected.q_desuperheat == pytest.approx(0.0)
    assert injected.q_liquid_injection == pytest.approx(dry.q_desuperheat)
    assert injected.q_latent_condense > dry.q_latent_condense
    assert injected.Q_cond == pytest.approx(dry.Q_cond)
    assert injected.Q_cond == pytest.approx(
        injected.m_dot * (injected.Hs[3] - injected.Hs[2])
    )


def test_mvr_liquid_injection_is_post_compression_desuperheating():
    pytest.importorskip("CoolProp")
    dry = MechanicalVapourRecompressionCycle()
    injected = MechanicalVapourRecompressionCycle()

    dry.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        eta_mvr_comp=0.75,
        liquid_injection=False,
    )
    injected.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        eta_mvr_comp=0.75,
        liquid_injection=True,
    )

    assert injected.source_m_dot == pytest.approx(dry.source_m_dot)
    assert injected.Hs[1] == pytest.approx(dry.Hs[1])
    assert injected.shaft_work == pytest.approx(dry.shaft_work)
    assert injected.m_dot > dry.m_dot
    assert injected.Q_cond == pytest.approx(dry.Q_cond)


def test_mvr_dry_desuperheat_respects_process_split():
    pytest.importorskip("CoolProp")
    no_process = MechanicalVapourRecompressionCycle()
    partial_process = MechanicalVapourRecompressionCycle()

    no_process.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        liquid_injection=False,
        process_split=0.0,
    )
    partial_process.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        liquid_injection=False,
        process_split=0.25,
    )

    assert no_process.solved is True
    assert partial_process.solved is True
    assert no_process.q_desuperheat > 0.0
    assert no_process.process_heat_components()["desuperheat"] == pytest.approx(0.0)
    assert no_process.process_heat == pytest.approx(0.0)
    assert no_process.process_m_dot_out == pytest.approx(no_process.m_dot)
    assert (
        len(
            list(
                no_process.build_stream_collection(
                    include_cond=True,
                    include_evap=False,
                )
            )
        )
        == 0
    )

    partial_components = partial_process.process_heat_components()
    assert partial_components["desuperheat"] == pytest.approx(
        partial_process.source_m_dot * 0.25 * partial_process.q_desuperheat
    )
    assert partial_process.process_m_dot_out == pytest.approx(
        partial_process.m_dot * 0.75
    )


def test_mvr_process_heat_components_rejects_invalid_split():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()
    mvr.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        liquid_injection=False,
    )

    with pytest.raises(ValueError, match="process_split"):
        mvr.process_heat_components(-0.1)
    with pytest.raises(ValueError, match="process_split"):
        mvr.process_heat_components(1.1)


def test_mvr_liquid_injection_mixer_conserves_enthalpy():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()

    mvr.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        eta_mvr_comp=0.75,
        liquid_injection=True,
    )

    assert mvr.solved is True
    h_injection_liq = mvr.Hs[2]
    injection_ratio = mvr.liquid_injection_ratio

    assert injection_ratio > 0.0
    assert mvr.Hs[1] + injection_ratio * h_injection_liq == pytest.approx(
        (1.0 + injection_ratio) * mvr.Hs[3],
        rel=1e-8,
    )
    assert mvr.q_liquid_injection == pytest.approx(
        injection_ratio * (mvr.Hs[3] - h_injection_liq),
        rel=1e-8,
    )


def test_mvr_source_heat_and_compressor_work_follow_enthalpy_balances():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()

    mvr.solve_from_source_heat(
        T_evap=85.0,
        T_cond=100.0,
        Q_source=1500.0,
        eta_mvr_comp=0.72,
        eta_motor=0.92,
    )

    assert mvr.solved is True
    h_low_liq = mvr._compute_state_from_pressure_quality(mvr.Ps[0], 0.0).hmass()
    assert mvr.q_source == pytest.approx(mvr.Hs[0] - h_low_liq)
    assert mvr.source_m_dot == pytest.approx(1500.0 / mvr.q_source)
    assert mvr.Q_evap == pytest.approx(1500.0)
    assert mvr.shaft_work == pytest.approx(mvr.source_m_dot * (mvr.Hs[1] - mvr.Hs[0]))
    assert mvr.work == pytest.approx(mvr.shaft_work / mvr.eta_motor)


def test_mvr_condenser_heat_uses_correct_mass_basis_with_and_without_injection():
    pytest.importorskip("CoolProp")
    dry = MechanicalVapourRecompressionCycle()
    injected = MechanicalVapourRecompressionCycle()

    dry.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        eta_mvr_comp=0.75,
        liquid_injection=False,
    )
    injected.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
        eta_mvr_comp=0.75,
        liquid_injection=True,
    )

    assert dry.solved is True
    assert injected.solved is True
    assert dry.Q_cond == pytest.approx(dry.source_m_dot * (dry.Hs[1] - dry.Hs[2]))
    assert injected.Q_cond == pytest.approx(
        injected.m_dot * (injected.Hs[3] - injected.Hs[2])
    )


def test_mvr_invalid_lift_returns_finite_unsolved_work():
    mvr = MechanicalVapourRecompressionCycle()

    work = mvr.solve_from_mass_flow(T_evap=120.0, T_cond=80.0, m_dot=0.01)

    assert mvr.solved is False
    assert np.isfinite(work)
    assert mvr.work == pytest.approx(work)


def test_mvr_invalid_state_returns_finite_unsolved_work():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()

    work = mvr.solve_from_mass_flow(T_evap=370.0, T_cond=380.0, m_dot=0.01)

    assert mvr.solved is False
    assert np.isfinite(work)
    assert mvr.work == pytest.approx(work)


def test_mvr_stream_collection_splits_condenser_and_evaporator_profiles():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()
    mvr.solve_from_source_heat(
        T_evap=80.0,
        T_cond=95.0,
        Q_source=1000.0,
    )

    cond_streams = mvr.build_stream_collection(include_cond=True, include_evap=False)
    evap_streams = mvr.build_stream_collection(include_cond=False, include_evap=True)

    assert cond_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        mvr.process_heat,
    )
    assert evap_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        mvr.Q_evap,
        rel=2e-2,
    )


def test_mvr_stream_collection_returns_only_process_heating_streams():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()
    mvr.solve_from_source_heat(
        T_evap=90.0,
        T_cond=105.0,
        Q_source=1000.0,
        dT_subcool=4.0,
        process_split=0.35,
    )

    cond_streams = mvr.build_stream_collection(include_cond=True, include_evap=False)

    assert mvr.process_heat < mvr.Q_cond
    assert mvr.process_m_dot_out > 0.0
    assert cond_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        mvr.process_heat
    )
    assert cond_streams.sum_stream_attribute("heat_flow") != pytest.approx(mvr.Q_cond)


def test_mvr_open_stage_stream_collection_uses_source_profile():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()
    mvr.solve_from_source_heat(
        T_evap=90.0,
        T_cond=105.0,
        Q_source=1000.0,
        dT_subcool=4.0,
    )

    evap_streams = mvr.build_stream_collection(include_cond=False, include_evap=True)

    assert evap_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        mvr.Q_evap,
        rel=2e-2,
    )


def test_mvr_mass_flow_stream_collection_does_not_emit_serial_source_heat():
    pytest.importorskip("CoolProp")
    mvr = MechanicalVapourRecompressionCycle()
    mvr.solve_from_mass_flow(
        T_evap=90.0,
        T_cond=105.0,
        m_dot=0.01,
        dT_subcool=4.0,
    )

    evap_streams = mvr.build_stream_collection(include_cond=False, include_evap=True)

    assert len(evap_streams) == 0


def test_vc_mvr_cascade_routes_split_and_excludes_internal_streams():
    pytest.importorskip("CoolProp")
    cascade = VapourCompressionMvrCascade()

    cascade.solve(
        T_evap_vc=np.array([20.0, 30.0]),
        T_cond_vc=np.array([90.0, 80.0]),
        dT_lift_mvr=np.array([15.0, 12.0]),
        Q_heat_vc=np.array([1000.0, 500.0]),
        mvr_source_split=0.25,
        mvr_process_split=np.array([0.4]),
        dT_subcool_vc=np.array([0.0, 0.0]),
        dT_subcool_mvr=np.array([4.0, 3.0]),
        dT_ihx_gas_side_vc=np.array([0.0, 0.0]),
        eta_comp=0.75,
        eta_mvr_comp=0.75,
        eta_motor=0.9,
        refrigerant=["R134A", "R134A"],
        mvr_fluid=["Water", "Water"],
        dt_cascade_hx=5.0,
    )

    assert cascade.solved is True
    assert cascade.mvr_cycles[0].dT_superheat == pytest.approx(0.0)
    assert cascade.mvr_cycles[0].dT_subcool == pytest.approx(4.0)
    np.testing.assert_allclose(cascade.internal_heat, np.array([250.0, 0.0]))
    np.testing.assert_allclose(cascade.direct_vc_heat, np.array([750.0, 500.0]))
    expected_mvr_evap_0 = cascade.vc_cycles[0].Ts[4] - 273.15 - 5.0
    np.testing.assert_allclose(
        cascade.T_evap_mvr,
        np.array([expected_mvr_evap_0, expected_mvr_evap_0 + 15.0]),
    )
    np.testing.assert_allclose(cascade.T_cond_mvr, cascade.T_evap_mvr + [15.0, 12.0])
    assert cascade.mvr_stage_mass_out[0] == pytest.approx(cascade.mvr_stage_mass_in[1])
    assert cascade.mvr_cycles[0].m_dot > cascade.mvr_stage_mass_in[0]
    assert cascade.mvr_stage_mass_out[-1] == pytest.approx(0.0)
    assert cascade.Q_heat_arr[-1] > 0.0
    assert cascade.COP_h == pytest.approx(cascade.Q_heat / cascade.work)

    external_streams = cascade.build_stream_collection(
        include_cond=True,
        include_evap=True,
    )
    internal_streams = cascade.build_stream_collection(
        include_cond=True,
        include_evap=True,
        include_internal=True,
    )
    assert external_streams.sum_stream_attribute(
        "heat_flow"
    ) < internal_streams.sum_stream_attribute("heat_flow")
    mvr_process_streams = [
        stream for stream in external_streams if str(stream.name).startswith("MVR_")
    ]
    assert not any(
        str(stream.name).startswith("MVR_desuperheat") for stream in mvr_process_streams
    )
    assert any(
        str(stream.name).startswith("MVR_condense") for stream in mvr_process_streams
    )
    assert any(
        str(stream.name).startswith("MVR_subcool") for stream in mvr_process_streams
    )
    assert sum(stream.heat_flow.value for stream in mvr_process_streams) == (
        pytest.approx(cascade.mvr_stage_heat.sum())
    )


def test_vc_mvr_cascade_propagates_post_injection_mass_and_component_heat():
    pytest.importorskip("CoolProp")
    cascade = VapourCompressionMvrCascade()

    cascade.solve(
        T_evap_vc=np.array([20.0]),
        T_cond_vc=np.array([90.0]),
        dT_lift_mvr=np.array([15.0, 12.0]),
        Q_heat_vc=np.array([1000.0]),
        mvr_source_split=0.25,
        mvr_process_split=np.array([0.35]),
        dT_subcool_vc=np.array([0.0]),
        dT_subcool_mvr=np.array([4.0, 3.0]),
        eta_comp=0.75,
        eta_mvr_comp=0.75,
        eta_motor=0.9,
        refrigerant=["R134A"],
        mvr_fluid=["Water", "Water"],
        dt_cascade_hx=5.0,
    )

    assert cascade.solved is True
    for j, cycle in enumerate(cascade.mvr_cycles):
        split = cascade.process_split[j]
        components = cycle.process_heat_components()
        assert cycle.process_split == pytest.approx(split)
        assert cascade.mvr_stage_mass_in[j] == pytest.approx(cycle.source_m_dot)
        assert cascade.mvr_stage_mass_out[j] == pytest.approx(cycle.process_m_dot_out)
        assert cycle.process_m_dot_out == pytest.approx(cycle.m_dot * (1.0 - split))
        assert cascade.mvr_stage_heat[j] == pytest.approx(components["total"])
        assert cascade._mvr_stage_latent_heat[j] == pytest.approx(components["latent"])
        assert cascade._mvr_stage_subcool_heat[j] == pytest.approx(
            components["subcool"]
        )

    assert cascade.mvr_stage_mass_in[1] == pytest.approx(cascade.mvr_stage_mass_out[0])
    assert cascade.mvr_stage_mass_out[-1] == pytest.approx(0.0)


def test_vc_mvr_cascade_infeasible_ordering_returns_finite_penalty():
    cascade = VapourCompressionMvrCascade()

    work = cascade.solve(
        T_evap_vc=np.array([60.0]),
        T_cond_vc=np.array([80.0]),
        dT_lift_mvr=np.array([25.0]),
        Q_heat_vc=np.array([1000.0]),
        mvr_source_split=0.5,
        dt_cascade_hx=5.0,
    )

    assert cascade.solved is False
    assert np.isfinite(work)
    assert any(penalty > 0.0 for penalty in cascade.penalty)


def test_vc_mvr_cascade_propagates_unsolved_vc_child_cycle(monkeypatch):
    class _UnsolvedCycle:
        solved = False
        work = -5.0

        def solve(self, **kwargs):
            return self.work

    monkeypatch.setattr(vcmvr_cascade_mod, "VapourCompressionCycle", _UnsolvedCycle)

    cascade = VapourCompressionMvrCascade()
    work = cascade.solve(
        T_evap_vc=np.array([20.0]),
        T_cond_vc=np.array([80.0]),
        dT_lift_mvr=np.array([10.0]),
        Q_heat_vc=np.array([1000.0]),
        mvr_source_split=0.25,
        dT_subcool_vc=np.array([0.0]),
        dT_subcool_mvr=np.array([0.0]),
        dT_ihx_gas_side_vc=np.array([0.0]),
        refrigerant=["R134A"],
        mvr_fluid=["Water"],
    )

    assert cascade.solved is False
    assert work > 0.0
    assert any(penalty > 0.0 for penalty in cascade.penalty)


def test_vc_mvr_x0_bounds_parse_round_trip():
    args = _base_args(
        n_cond=2,
        n_evap=2,
        n_mvr=2,
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
    )
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=20.0,
    )

    x0, bnds = hp_vc_mvr._get_vc_mvr_opt_setup(init_res=init_res, args=args)
    state = hp_vc_mvr._parse_vc_mvr_state_variables(x0, args)

    assert x0.shape == (18,)
    assert len(bnds) == x0.shape[0]
    unpacked = hp_vc_mvr._unpack_vc_mvr_state(state, args)
    np.testing.assert_allclose(unpacked["T_cond_vc"], init_res.T_cond)
    np.testing.assert_allclose(unpacked["T_evap_vc"], init_res.T_evap)
    assert state.Q_heat_base == pytest.approx(init_res.Q_cond.sum())
    np.testing.assert_allclose(
        decode_duty_splits(state.x_heat_split, state.Q_heat_base),
        init_res.Q_cond,
    )
    assert state.Q_heat_available.shape == init_res.Q_cond.shape
    np.testing.assert_allclose(state.T_cond[:2], unpacked["T_cond_mvr"])
    np.testing.assert_allclose(state.T_cond[2:], init_res.T_cond)
    np.testing.assert_allclose(state.T_evap[:2], unpacked["T_evap_mvr"])
    np.testing.assert_allclose(state.T_evap[2:], init_res.T_evap)
    assert state.dT_subcool.shape == (4,)
    np.testing.assert_allclose(state.dT_subcool[:2], unpacked["dT_subcool_mvr"])
    np.testing.assert_allclose(state.dT_subcool[2:], unpacked["dT_subcool_vc"])
    assert unpacked["T_evap_mvr"].shape == (2,)
    assert unpacked["T_cond_mvr"].shape == (2,)
    assert unpacked["mvr_source_split"] == pytest.approx(0.5)
    assert unpacked["mvr_process_split"].shape == (1,)
    np.testing.assert_allclose(
        unpacked["T_evap_mvr"][0],
        unpacked["T_cond_vc"][0] - unpacked["dT_subcool_vc"][0] - args.dt_cascade_hx,
    )
    np.testing.assert_allclose(unpacked["T_evap_mvr"][1], unpacked["T_cond_mvr"][0])
    assert np.all(unpacked["dT_lift_mvr"] <= 20.0)
    assert state.Q_amb_cold == pytest.approx(init_res.Q_amb_cold)
    assert state.Q_amb_cold_direct == pytest.approx(init_res.Q_amb_cold)
    assert state.Q_amb_cold_residual == pytest.approx(0.0)


def test_vc_mvr_heat_duties_use_split_fractions():
    args = _base_args(n_cond=3, n_evap=3, n_mvr=1)
    x = np.ones(hp_vc_mvr._vc_mvr_layout(args).size) * 0.5
    state = hp_vc_mvr._parse_vc_mvr_state_variables(x, args)

    Q_heat_request = decode_duty_splits(state.x_heat_split, state.Q_heat_base)

    assert Q_heat_request.sum() < args.Q_heat_max + state.Q_amb_cold
    np.testing.assert_allclose(
        state.x_heat_split,
        np.array([0.5, 0.5, 0.5]),
    )


def test_vc_mvr_rejects_zero_mvr_count():
    args = _base_args(n_mvr=0)

    with pytest.raises(ValueError, match="at least one MVR"):
        hp_vc_mvr._vc_mvr_layout(args)


def test_compute_vc_mvr_system_obj_solved(monkeypatch):
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=1,
        refrigerant_ls=["R134A"],
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
    )
    x = np.array([0.2] * hp_vc_mvr._vc_mvr_layout(args).size)

    class _FakeCascadeSolved:
        solved = True
        work = 40.0
        work_arr = np.array([25.0, 15.0])
        Q_heat = 200.0
        Q_heat_arr = np.array([80.0, 120.0])
        Q_cool_arr = np.array([90.0, 0.0])
        penalty = [0.0]
        T_cond_mvr = np.array([100.0])
        T_evap_mvr = np.array([90.0])
        mvr_cycles = [SimpleNamespace(dT_subcool=1.0)]
        vc_cycles = [SimpleNamespace(dT_subcool=2.0)]

        def solve(self, **kwargs):
            self.kwargs = kwargs

        def build_stream_collection(self, **kwargs):
            return _sc(_stream("HP", 100.0, 90.0, 10.0, is_process_stream=False))

    monkeypatch.setattr(hp_vc_mvr, "VapourCompressionMvrCascade", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(5.0, 1.0), _pt_with_hnet(6.0, 2.0)])
    monkeypatch.setattr(
        hp_shared,
        "get_process_heat_cascade",
        lambda **kwargs: next(seq),
    )

    out = hp_vc_mvr._compute_vc_mvr_system_obj(x, args)

    assert out.success is True
    assert out.w_net == pytest.approx(40.0)
    np.testing.assert_allclose(out.w_hpr, np.array([15.0, 25.0]))
    np.testing.assert_allclose(out.T_evap[:1], np.array([90.0]))
    np.testing.assert_allclose(out.T_cond[:1], np.array([100.0]))
    assert out.cop_h == pytest.approx(5.0)
    np.testing.assert_allclose(out.Q_heat, np.array([120.0, 80.0]))


def test_compute_vc_mvr_system_obj_real_cascade_smoke():
    pytest.importorskip("CoolProp")
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=1,
        refrigerant_ls=["R134A"],
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
        initialise_simulated_cycle=False,
    )
    x = np.array([0.0, 0.6, 0.1, 0.1, 0.1, 0.5, 0.5, 0.0, 0.5, 0.5])

    out = hp_vc_mvr._compute_vc_mvr_system_obj(x, args, debug=True)

    assert out.success is True
    assert isinstance(out.model, VapourCompressionMvrCascade)
    assert out.w_net > 0.0
    assert out.Q_heat.sum() > 0.0
    assert out.Q_cool.sum() > 0.0
    assert out.hpr_streams is not None
    assert out.hpr_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        out.Q_heat.sum() + out.Q_cool.sum()
    )


def test_compute_vc_mvr_system_obj_debug_reraises(monkeypatch):
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=1,
        refrigerant_ls=["R134A"],
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
    )
    x = np.array([0.2] * hp_vc_mvr._vc_mvr_layout(args).size)

    class _BrokenCascade:
        def solve(self, **kwargs):
            raise RuntimeError("cascade exploded")

    monkeypatch.setattr(hp_vc_mvr, "VapourCompressionMvrCascade", _BrokenCascade)

    out = hp_vc_mvr._compute_vc_mvr_system_obj(x, args)

    assert out.success is False
    assert "cascade exploded" in out.failure_reason
    with pytest.raises(RuntimeError, match="cascade exploded"):
        hp_vc_mvr._compute_vc_mvr_system_obj(x, args, debug=True)


def test_vc_mvr_refrigeration_rejected_clearly():
    args = _base_args(
        is_heat_pumping=False,
        n_mvr=1,
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
    )

    with pytest.raises(ValueError, match="heat-pump-only"):
        hp_vc_mvr.optimise_vapour_compression_mvr_heat_pump_placement(args)


def test_vc_mvr_config_schema_and_dispatch(monkeypatch):
    config = Configuration(
        {
            "HPR_TYPE": HPRcycle.VapourCompMVR.value,
            "N_MVR": 2,
            "MVR_FLUIDS": "Water;R134A",
            "ETA_MVR_COMP": 0.65,
        }
    )
    args = construct_HPRTargetInputs(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -5.0, -10.0]),
        H_cold=np.array([10.0, 5.0, 0.0]),
        zone_config=config,
    )

    assert args.n_mvr == 2
    assert args.mvr_fluid_ls == ["Water", "R134A"]
    assert args.eta_mvr_comp == pytest.approx(0.65)

    _patch_output_model_validate(monkeypatch)
    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda **kwargs: SimpleNamespace(hpr_type=HPRcycle.VapourCompMVR.value),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        HPRcycle.VapourCompMVR.value,
        lambda args: HPRBackendResult(
            obj=0.1,
            utility_tot=1.0,
            w_net=0.5,
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            artifacts=HPRThermoArtifacts(hpr_streams=_sc()),
        ),
    )

    out = hp._get_hpr_targets(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        zone_config=SimpleNamespace(HPR_TYPE=HPRcycle.VapourCompMVR.value),
        is_heat_pumping=True,
    )

    assert out["success"] is True
