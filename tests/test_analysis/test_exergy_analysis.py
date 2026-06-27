"""Regression tests for exergy targeting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import GT, PT, TT
from OpenPinch.lib.schemas.targets import (
    DirectHeatPumpTarget,
    DirectIntegrationTarget,
    IndirectHeatPumpTarget,
    TotalSiteTarget,
)
from OpenPinch.services.exergy_analysis import (
    apply_exergy_if_enabled,
    apply_exergy_targeting,
    build_exergy_gcc_curve,
    build_exergy_nlp_curves,
    compute_exergetic_temperature,
)
from OpenPinch.services.exergy_analysis.exergy_targeting_entry import (
    _first_available_column,
    _insert_breaks,
    _make_target_spec,
    _normalize_exergy_base_target_type,
    _optional_column,
    _resolve_target_exergy_spec,
    _should_use_series,
    _strictly_between,
    _validate_curve_lengths,
)


def _make_problem_table(columns: dict) -> ProblemTable:
    return ProblemTable(columns)


def _base_config() -> Configuration:
    config = Configuration()
    config.environment.temperature = 15.0
    config.thermal.dt_cont = 10.0
    return config


def _di_problem_table() -> ProblemTable:
    return _make_problem_table(
        {
            PT.T: [80.0, 40.0, 10.0, -20.0],
            PT.H_NET_A: [0.0, 20.0, 35.0, 55.0],
            PT.H_NET_HOT: [30.0, 20.0, 8.0, 0.0],
            PT.H_NET_COLD: [0.0, 8.0, 18.0, 26.0],
        }
    )


def _ts_problem_table() -> ProblemTable:
    return _make_problem_table(
        {
            PT.T: [120.0, 60.0, 20.0, -10.0],
            PT.H_NET_UT: [0.0, 15.0, 28.0, 40.0],
            PT.H_HOT_UT: [22.0, 16.0, 5.0, 0.0],
            PT.H_COLD_UT: [0.0, 6.0, 15.0, 24.0],
        }
    )


def _dhp_problem_table() -> ProblemTable:
    return _make_problem_table(
        {
            PT.T: [90.0, 45.0, 20.0, -15.0],
            PT.H_NET_W_AIR: [0.0, 18.0, 30.0, 42.0],
            PT.H_NET_HOT: [28.0, 18.0, 6.0, 0.0],
            PT.H_NET_COLD: [0.0, 7.0, 16.0, 24.0],
            PT.H_HOT_UT: [12.0, 9.0, 2.0, 0.0],
            PT.H_COLD_UT: [0.0, 2.0, 5.0, 9.0],
            PT.H_HOT_HP: [8.0, 6.0, 2.0, 0.0],
            PT.H_COLD_HP: [0.0, 1.0, 4.0, 8.0],
        }
    )


def _ihp_problem_table() -> ProblemTable:
    return _make_problem_table(
        {
            PT.T: [110.0, 55.0, 15.0, -25.0],
            PT.H_NET_UT: [0.0, 14.0, 26.0, 38.0],
            PT.H_NET_HP: [0.0, 6.0, 12.0, 18.0],
            PT.H_HOT_UT: [18.0, 12.0, 4.0, 0.0],
            PT.H_COLD_UT: [0.0, 5.0, 11.0, 18.0],
            PT.H_HOT_HP: [10.0, 7.0, 3.0, 0.0],
            PT.H_COLD_HP: [0.0, 2.0, 6.0, 10.0],
        }
    )


def _make_direct_integration_target() -> DirectIntegrationTarget:
    config = _base_config()
    return DirectIntegrationTarget(
        zone_name="Plant",
        type=TT.DI.value,
        config=config,
        pt=_di_problem_table(),
        pt_real=_di_problem_table(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
    )


def _make_total_site_target() -> TotalSiteTarget:
    config = _base_config()
    return TotalSiteTarget(
        zone_name="Plant",
        type=TT.TS.value,
        config=config,
        pt=_ts_problem_table(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
    )


def _make_direct_heat_pump_target() -> DirectHeatPumpTarget:
    config = _base_config()
    return DirectHeatPumpTarget(
        zone_name="Plant",
        type=TT.DHP.value,
        config=config,
        pt=_dhp_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
        hpr_cycle="stub",
        hpr_utility_total=0.0,
        hpr_work=0.0,
        hpr_external_utility=0.0,
        hpr_ambient_hot=0.0,
        hpr_ambient_cold=0.0,
        hpr_cop=0.0,
        hpr_eta_he=0.0,
        hpr_success=True,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details={},
    )


def _make_indirect_heat_pump_target() -> IndirectHeatPumpTarget:
    config = _base_config()
    return IndirectHeatPumpTarget(
        zone_name="Plant",
        type=TT.IHP.value,
        config=config,
        pt=_ihp_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
        hpr_cycle="stub",
        hpr_utility_total=0.0,
        hpr_work=0.0,
        hpr_external_utility=0.0,
        hpr_ambient_hot=0.0,
        hpr_ambient_cold=0.0,
        hpr_cop=0.0,
        hpr_eta_he=0.0,
        hpr_success=True,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details={},
    )


def test_compute_exergetic_temperature_handles_ambient_and_both_sides():
    assert compute_exergetic_temperature(15.0, T_ref_in_C=15.0) == pytest.approx(0.0)
    assert compute_exergetic_temperature(80.0, T_ref_in_C=15.0) > 0.0
    assert compute_exergetic_temperature(-20.0, T_ref_in_C=15.0) > 0.0


def test_build_exergy_gcc_curve_splits_segments_that_cross_ambient():
    payload = build_exergy_gcc_curve(
        temperatures=[60.0, 20.0, -10.0],
        heat_loads=[0.0, 15.0, 28.0],
        t_env=15.0,
        dt_cont_half=0.0,
    )

    assert 0.0 in payload[PT.T.value]
    assert len(payload[PT.T.value]) > 3
    assert min(payload[PT.X_GCC.value]) >= 0.0


def test_build_exergy_gcc_curve_handles_zero_negative_and_single_point_segments():
    duplicate_temperature = build_exergy_gcc_curve(
        temperatures=[100.0, 100.0, 50.0],
        heat_loads=[0.0, 0.0, 10.0],
        t_env=15.0,
    )
    assert len(duplicate_temperature[PT.T.value]) == 2

    sign_change = build_exergy_gcc_curve(
        temperatures=[100.0, 60.0, 20.0, -20.0],
        heat_loads=[0.0, 20.0, 5.0, 5.0],
        t_env=15.0,
        dt_cont_half=5.0,
    )
    assert len(sign_change[PT.T.value]) > 4
    assert min(sign_change[PT.X_GCC.value]) == pytest.approx(0.0)

    one_point = build_exergy_gcc_curve(
        temperatures=[15.0],
        heat_loads=[0.0],
        t_env=15.0,
    )
    assert one_point == {PT.T.value: [0.0], PT.X_GCC.value: [0.0]}


def test_build_exergy_nlp_curves_handles_empty_constant_and_cross_ambient_branches():
    empty = build_exergy_nlp_curves(
        temperatures=[60.0, 20.0, -20.0],
        branches=[("hot", [0.0, 0.0, 0.0]), ("cold", [np.nan, np.nan, np.nan])],
        t_env=15.0,
    )
    assert empty["source_total"] == 0.0
    assert empty["sink_total"] == 0.0

    mixed = build_exergy_nlp_curves(
        temperatures=[60.0, 20.0, -20.0],
        branches=[
            ("hot", [10.0, 10.0, 10.0]),
            ("hot", [0.0, 12.0, 24.0]),
            ("cold", [0.0, 8.0, 18.0]),
        ],
        t_env=15.0,
    )
    assert mixed["source_total"] > 0.0
    assert mixed["sink_total"] > 0.0


@pytest.mark.parametrize(
    "factory",
    [
        _make_direct_integration_target,
        _make_total_site_target,
        _make_direct_heat_pump_target,
        _make_indirect_heat_pump_target,
    ],
)
def test_apply_exergy_targeting_populates_graphs_and_scalar_targets(factory):
    target = factory()

    apply_exergy_targeting(target)

    assert GT.GCC_X.value in target.graphs
    assert GT.NLP_X.value in target.graphs
    assert PT.X_GCC.value in target.graphs[GT.GCC_X.value]
    assert PT.X_SUR.value in target.graphs[GT.NLP_X.value]
    assert PT.X_DEF.value in target.graphs[GT.NLP_X.value]
    assert target.exergy_sources is not None
    assert target.exergy_sinks is not None
    assert target.ETE is not None
    assert target.exergy_req_min is not None
    assert target.exergy_des_min is not None


def test_apply_exergy_targeting_returns_unsupported_targets_unchanged():
    target = SimpleNamespace(type="unsupported", graphs={})

    assert apply_exergy_targeting(target) is target


def test_apply_exergy_if_enabled_uses_grouped_targeting_config_only():
    config = Configuration()
    zone = SimpleNamespace(config=config)
    target = object()
    calls = []

    def apply_func(value):
        calls.append(value)
        return "applied"

    assert apply_exergy_if_enabled(target, zone, apply_func=apply_func) is target
    assert calls == []

    config.targeting.exergy_enabled = True

    assert apply_exergy_if_enabled(target, zone, apply_func=apply_func) == "applied"
    assert calls == [target]
    assert not hasattr(config, "TARGETING_EXERGY_ENABLED")


def test_total_site_serialization_includes_exergy_fields_after_enrichment():
    target = _make_total_site_target()

    apply_exergy_targeting(target)
    payload = target.serialize_json()

    assert payload["exergy_sources"] == {"value": target.exergy_sources, "unit": "kW"}
    assert payload["exergy_sinks"] == {"value": target.exergy_sinks, "unit": "kW"}
    assert payload["ETE"] == {"value": target.ETE * 100, "unit": "%"}
    assert payload["exergy_req_min"] == {"value": target.exergy_req_min, "unit": "kW"}
    assert payload["exergy_des_min"] == {"value": target.exergy_des_min, "unit": "kW"}


def test_exergy_target_selection_and_column_helpers_cover_edge_cases():
    with pytest.raises(ValueError, match="Unsupported exergy base_target_type"):
        _normalize_exergy_base_target_type("unsupported")

    for target_type in (TT.DI.value, TT.TS.value, TT.DHP.value, TT.IHP.value):
        assert (
            _resolve_target_exergy_spec(SimpleNamespace(type=target_type, pt=None))
            is None
        )
    assert _resolve_target_exergy_spec(SimpleNamespace(type="unsupported")) is None

    assert (
        _make_target_spec(
            temperatures=[100.0, 50.0],
            gcc_series=None,
            branches=[],
            t_env=15.0,
            dt_cont_half=0.0,
        )
        is None
    )

    class MissingColumnTable:
        def __getitem__(self, column):
            raise KeyError(column)

    assert _optional_column(MissingColumnTable(), PT.H_NET) is None
    assert _optional_column({PT.H_NET: [np.nan, np.nan]}, PT.H_NET) is None
    assert _first_available_column(MissingColumnTable(), [PT.H_NET, PT.H_NET_A]) is None
    assert _should_use_series([np.nan, np.inf]) is False

    with pytest.raises(ValueError, match="matching lengths"):
        _validate_curve_lengths(np.array([1.0, 2.0]), np.array([1.0]))


def test_exergy_breakpoint_helpers_cover_short_and_ascending_profiles():
    assert _insert_breaks([25.0], [15.0]) == [25.0]
    assert _insert_breaks([0.0, 100.0], [50.0]) == [0.0, 50.0, 100.0]
    assert _strictly_between(50.0, 0.0, 100.0, descending=False) is True
    assert _strictly_between(50.0, 100.0, 0.0, descending=True) is True
