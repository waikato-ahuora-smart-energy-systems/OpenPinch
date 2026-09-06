"""Winning-record to performance-map basis compatibility contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.performance_maps.target_basis import (
    HprPerformanceMapCompatibilityError,
    build_hpr_target_map_basis,
)
from OpenPinch.domain.targets import (
    DirectHeatPumpTarget,
    DirectRefrigerationTarget,
    HeatPumpTargetBase,
    SubzoneAggregateTarget,
)
from tests.contracts.test_hpr_target_simulation_record import _record
from tests.strategies.hpr_targeting import hpr_target_simulation_records


def _target(record=None, **overrides):
    record = _record() if record is None else record
    target_cls = (
        DirectHeatPumpTarget
        if record.mode == "heat_pump"
        else DirectRefrigerationTarget
    )
    useful = record.nominal_useful_duty
    power = useful / 4.0
    details = SimpleNamespace(
        simulation_backend=record.simulation_backend,
        target_simulation_record=record,
        period_outputs=None,
        T_evap=np.array([record.nominal_evaporating_temperature]),
        T_cond=np.array([record.nominal_condensing_temperature]),
        Q_heat=np.array([useful if record.mode == "heat_pump" else useful + power]),
        Q_cool=np.array([useful - power if record.mode == "heat_pump" else useful]),
        dT_superheat=np.array([record.superheat]),
        dT_subcool=np.array([record.subcooling]),
    )
    values = {
        "name": "Plant/Direct HPR",
        "period_id": record.period_id,
        "hpr_success": True,
        "hpr_simulation_backend": record.simulation_backend,
        "hpr_details": details,
    }
    values.update(overrides)
    return target_cls.model_construct(**values)


def _replace_details(target, **updates):
    return SimpleNamespace(**(vars(target.hpr_details) | updates))


def test_basis_is_field_exact_idempotent_and_deep_detached() -> None:
    record = _record(
        assumptions={"anchor": "dew_bubble", "components": ["R32", "R125"]}
    )
    target = _target(record)
    before = record.model_dump(mode="json")

    first = build_hpr_target_map_basis(target)
    second = build_hpr_target_map_basis(target)

    assert first == second
    assert first is not second
    assert first.target_id == target.name
    assert first.simulation_backend == record.simulation_backend
    assert first.mode == record.mode
    assert first.cycle_id == record.cycle_id
    assert first.model_id == record.model_id
    assert first.refrigerant_spec == record.refrigerant_spec
    assert first.nominal_useful_duty == record.nominal_useful_duty
    assert first.source_provenance is not second.source_provenance
    first.source_provenance["assumptions"]["components"].append("mutated")
    assert record.model_dump(mode="json") == before


@pytest.mark.parametrize(
    ("target_factory", "code"),
    [
        (lambda: object(), "unsupported_target_type"),
        (
            lambda: SubzoneAggregateTarget.model_construct(name="aggregate"),
            "aggregate_target",
        ),
        (lambda: _target(hpr_success=False), "failed_target"),
        (
            lambda: _target(
                hpr_details=SimpleNamespace(
                    simulation_backend="coolprop",
                    target_simulation_record=None,
                    period_outputs=None,
                )
            ),
            "missing_simulation_record",
        ),
        (
            lambda: _target(_record(evaporator_count=2)),
            "multi_port_topology",
        ),
        (
            lambda: _target(_record().model_copy(update={"cycle_id": "other"})),
            "multi_port_topology",
        ),
        (
            lambda: _target(
                hpr_details=_replace_details(
                    _target(), target_simulation_record=object()
                )
            ),
            "record_inconsistency",
        ),
        (
            lambda: (
                lambda target: _target(
                    hpr_details=_replace_details(
                        target,
                        period_outputs={"base": object()},
                    )
                )
            )(_target()),
            "aggregate_target",
        ),
        (
            lambda: (
                lambda target: _target(
                    hpr_details=_replace_details(
                        target,
                        T_cond=np.array([55.0, 65.0]),
                    )
                )
            )(_target()),
            "non_scalar_nominal_data",
        ),
        (
            lambda: _target(hpr_simulation_backend="tespy"),
            "record_inconsistency",
        ),
        (
            lambda: (
                lambda target: _target(
                    hpr_details=_replace_details(target, T_evap="not-a-number")
                )
            )(_target()),
            "non_scalar_nominal_data",
        ),
        (
            lambda: (
                lambda target: _target(
                    hpr_details=_replace_details(target, T_evap=np.array([np.nan]))
                )
            )(_target()),
            "non_scalar_nominal_data",
        ),
        (
            lambda: (
                lambda target: _target(
                    hpr_details=_replace_details(target, T_evap=np.array([999.0]))
                )
            )(_target()),
            "record_inconsistency",
        ),
        (
            lambda: HeatPumpTargetBase.model_construct(
                name="unsupported-subtype",
                hpr_success=True,
                hpr_simulation_backend="coolprop",
                hpr_details=vars(_target().hpr_details),
            ),
            "unsupported_target_type",
        ),
    ],
)
def test_incompatible_target_contexts_raise_typed_codes(target_factory, code) -> None:
    with pytest.raises(HprPerformanceMapCompatibilityError) as captured:
        build_hpr_target_map_basis(target_factory())

    assert captured.value.code == code


@seed(20260715)
@given(
    context=st.sampled_from(
        (
            "unsupported",
            "aggregate",
            "failed",
            "missing_record",
            "multi_port",
            "period_aggregate",
            "non_scalar",
            "backend_mismatch",
        )
    )
)
def test_generated_incompatible_contexts_fail_before_simulation(context: str) -> None:
    factories = {
        "unsupported": lambda: object(),
        "aggregate": lambda: SubzoneAggregateTarget.model_construct(name="aggregate"),
        "failed": lambda: _target(hpr_success=False),
        "missing_record": lambda: _target(
            hpr_details=SimpleNamespace(
                simulation_backend="coolprop",
                target_simulation_record=None,
                period_outputs=None,
            )
        ),
        "multi_port": lambda: _target(_record(evaporator_count=2)),
        "period_aggregate": lambda: (
            lambda target: _target(
                hpr_details=_replace_details(
                    target,
                    period_outputs={"base": object()},
                )
            )
        )(_target()),
        "non_scalar": lambda: (
            lambda target: _target(
                hpr_details=_replace_details(
                    target,
                    T_cond=np.array([55.0, 65.0]),
                )
            )
        )(_target()),
        "backend_mismatch": lambda: _target(hpr_simulation_backend="tespy"),
    }

    with pytest.raises(HprPerformanceMapCompatibilityError):
        build_hpr_target_map_basis(factories[context]())


def test_basis_builder_never_reads_private_engine_model() -> None:
    base = _target()

    class DetailsWithoutModel(SimpleNamespace):
        @property
        def model(self):
            raise AssertionError("basis extraction must not inspect engine model")

    target = _target(hpr_details=DetailsWithoutModel(**vars(base.hpr_details)))

    assert build_hpr_target_map_basis(target).model_id == (
        target.hpr_details.target_simulation_record.model_id
    )


@seed(20260715)
@given(hpr_target_simulation_records())
def test_generated_records_preserve_backend_mode_fluid_and_nominal_values(
    record,
) -> None:
    target = _target(record)

    basis = build_hpr_target_map_basis(target)

    assert basis.simulation_backend == record.simulation_backend
    assert basis.mode == record.mode
    assert basis.refrigerant_spec == record.refrigerant_spec
    assert basis.nominal_evaporating_temperature == (
        record.nominal_evaporating_temperature
    )
    assert basis.nominal_condensing_temperature == record.nominal_condensing_temperature
    assert basis.nominal_useful_duty == record.nominal_useful_duty
