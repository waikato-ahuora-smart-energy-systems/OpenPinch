"""Backend-selector plumbing contracts for current HPR targeting methods."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import OpenPinch.analysis.heat_pumps.service as hp
from OpenPinch.analysis.heat_pumps.common.preprocessing import (
    construct_HPRTargetInputs,
)
from OpenPinch.application._problem.periods.aggregation import (
    _TargetRunSpec,
    target_output_for_recorded_period,
)
from OpenPinch.contracts.hpr import HPRBackendResult
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import (
    HeatPumpAndRefrigerationCycle,
    ProblemTableLabel,
    TargetType,
)
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import DirectHeatPumpTarget
from OpenPinch.presentation.reporting.results import target_to_result
from tests.analysis.heat_pumps.helpers import _base_args, _patch_output_model_validate
from tests.contracts.test_hpr_target_simulation_record import _record


def test_preprocessing_carries_normalized_backend_intent() -> None:
    args = construct_HPRTargetInputs(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -5.0, -20.0]),
        H_cold=np.array([20.0, 5.0, 0.0]),
        is_heat_pumping=True,
        config=Configuration(),
        simulation_backend="tespy",
    )

    assert args.simulation_backend == "tespy"


def test_service_passes_backend_to_preprocessing_and_output(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_output_model_validate(monkeypatch)

    def fake_construct(*_args, simulation_backend="coolprop", **_kwargs):
        captured["simulation_backend"] = simulation_backend
        return _base_args(
            hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
            simulation_backend=simulation_backend,
            n_cond=1,
            n_evap=1,
            allow_integrated_expander=False,
        )

    monkeypatch.setattr(hp, "construct_HPRTargetInputs", fake_construct)
    monkeypatch.setattr(
        hp,
        "preflight_tespy_hpr_targeting",
        lambda args: captured.__setitem__("preflight_backend", args.simulation_backend),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        lambda args, *, prepared_tespy: HPRBackendResult.failure().with_updates(
            simulation_backend=args.simulation_backend
        ),
    )

    result = hp._get_hpr_targets(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        config=Configuration(),
        is_heat_pumping=True,
        simulation_backend="tespy",
    )

    assert captured == {
        "simulation_backend": "tespy",
        "preflight_backend": "tespy",
    }
    assert result["simulation_backend"] == "tespy"


def test_domain_target_defaults_to_coolprop_and_accepts_tespy() -> None:
    common = {
        "zone_name": "Site",
        "scope": "Site",
        "type": TargetType.DHP.value,
        "pt": ProblemTable({ProblemTableLabel.T: [100.0]}),
        "hpr_cycle": "Cascade vapour compression cycles",
        "hpr_utility_total": 1.0,
        "hpr_work": 1.0,
        "hpr_external_utility": 0.0,
        "hpr_ambient_hot": 0.0,
        "hpr_ambient_cold": 0.0,
        "hpr_cop": 3.0,
        "hpr_eta_he": 0.0,
        "hpr_success": True,
        "hpr_hot_streams": StreamCollection(),
        "hpr_cold_streams": StreamCollection(),
        "hpr_details": {},
    }

    default = DirectHeatPumpTarget.model_validate(common)
    record = _record(simulation_backend="tespy", period_id=None)
    selected = DirectHeatPumpTarget.model_validate(
        common
        | {
            "hpr_simulation_backend": "tespy",
            "hpr_details": SimpleNamespace(target_simulation_record=record),
        }
    )

    assert default.hpr_simulation_backend == "coolprop"
    assert selected.hpr_simulation_backend == "tespy"
    reported = target_to_result(selected)
    assert reported.hpr_simulation_backend == "tespy"
    assert reported.hpr_target_simulation_record == record


def test_summary_adds_backend_provenance() -> None:
    result = HPRBackendResult.failure().with_updates(simulation_backend="tespy")
    zone = SimpleNamespace(
        config=SimpleNamespace(
            hpr=SimpleNamespace(
                type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value
            )
        )
    )

    assert hp._get_hpr_target_summary(result, zone)["hpr_simulation_backend"] == "tespy"


def test_scalar_summary_attaches_period_to_a_detached_winning_record() -> None:
    record = _record(period_id=None)
    result = HPRBackendResult.failure().with_updates(
        success=True,
        simulation_backend="coolprop",
        target_simulation_record=record,
    )
    zone = SimpleNamespace(
        config=SimpleNamespace(
            hpr=SimpleNamespace(
                type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value
            )
        )
    )

    summary = hp._get_hpr_target_summary(result, zone, period_id="winter")

    assert summary["hpr_details"].target_simulation_record.period_id == "winter"
    assert result.target_simulation_record.period_id is None


def test_recorded_period_replay_preserves_backend_and_zone_keyword() -> None:
    captured: dict[str, object] = {}
    output = TargetOutput(name="Site", targets=[])

    def target_method(*, zone, options, include_subzones):
        captured.update(
            zone=zone,
            options=options,
            include_subzones=include_subzones,
        )

    problem = SimpleNamespace(
        target=SimpleNamespace(vapour_compression_heat_pump=target_method),
        _results=output,
    )
    spec = _TargetRunSpec(
        surface="vapour_compression_heat_pump",
        options={"simulation_backend": "tespy", "period_id": "old"},
        zone_name="AreaA",
        include_subzones=True,
    )

    replayed = target_output_for_recorded_period(problem, spec, "peak")

    assert replayed == output
    assert captured == {
        "zone": "AreaA",
        "options": {"simulation_backend": "tespy", "period_id": "peak"},
        "include_subzones": True,
    }
