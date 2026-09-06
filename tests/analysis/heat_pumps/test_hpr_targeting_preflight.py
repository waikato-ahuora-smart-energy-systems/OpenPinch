"""TESPy targeting compatibility preflight contracts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

import OpenPinch.analysis.heat_pumps.service as hp
from OpenPinch.analysis.heat_pumps.performance_maps.targeting import (
    HprTargetCompatibilityError,
    PreparedTespyHprTargeting,
    normalize_hpr_simulation_backend,
    preflight_tespy_hpr_targeting,
)
from OpenPinch.contracts.hpr import HeatPumpTargetInputs, MultiPeriodHPRTargetInputs
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from tests.analysis.heat_pumps.helpers import _base_args
from tests.strategies.hpr_targeting import BACKENDS, WORKING_FLUIDS


def _target_args(**overrides) -> HeatPumpTargetInputs:
    values = {
        "hpr_type": HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        "n_cond": 1,
        "n_evap": 1,
        "refrigerant_ls": ["R134a"],
        "allow_integrated_expander": False,
        "dt_hp_ihx": 0.0,
        "simulation_backend": "tespy",
        "T_hot": np.array([5.0, 0.0, -20.0]),
        "T_cold": np.array([100.0, 80.0, 55.0]),
    }
    values.update(overrides)
    data = vars(_base_args(**values)).copy()
    data.setdefault("n_mvr", 1)
    data.setdefault("eta_mvr_comp", 0.7)
    data.setdefault("eta_motor", 0.95)
    data.setdefault("mvr_fluid_ls", ["Water"])
    return HeatPumpTargetInputs.model_validate(data)


@seed(20260715)
@given(backend=BACKENDS)
def test_backend_normalization_is_idempotent(backend: str) -> None:
    normalized = normalize_hpr_simulation_backend(backend)

    assert normalize_hpr_simulation_backend(normalized) == normalized


@pytest.mark.parametrize("value", [None, 1, object()])
def test_backend_normalization_rejects_non_strings(value) -> None:
    with pytest.raises(TypeError):
        normalize_hpr_simulation_backend(value)


def test_backend_normalization_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        normalize_hpr_simulation_backend("refprop")


@pytest.mark.parametrize("is_heat_pumping", [True, False])
def test_preflight_accepts_approved_single_stage_modes(is_heat_pumping: bool) -> None:
    prepared = preflight_tespy_hpr_targeting(
        _target_args(is_heat_pumping=is_heat_pumping)
    )

    assert isinstance(prepared, PreparedTespyHprTargeting)
    assert prepared.mode == ("heat_pump" if is_heat_pumping else "refrigeration")
    assert prepared.cycle_id == "single_stage_vapour_compression"
    assert prepared.condenser_count == prepared.evaporator_count == 1
    assert prepared.working_fluid.source_spec == "R134a"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"n_cond": 2}, "unsupported_topology"),
        ({"n_evap": 2}, "unsupported_topology"),
        (
            {
                "hpr_type": HeatPumpAndRefrigerationCycle.ParallelVapourComp.value,
            },
            "unsupported_cycle",
        ),
        (
            {"hpr_type": HeatPumpAndRefrigerationCycle.CascadeCarnot.value},
            "unsupported_cycle",
        ),
        (
            {"hpr_type": HeatPumpAndRefrigerationCycle.Brayton.value},
            "unsupported_cycle",
        ),
        (
            {"hpr_type": HeatPumpAndRefrigerationCycle.VapourCompMVR.value},
            "unsupported_cycle",
        ),
        ({"allow_integrated_expander": True}, "integrated_expander_unsupported"),
        ({"dt_hp_ihx": 1.0}, "unsupported_model"),
    ],
)
def test_preflight_rejects_unsupported_cycle_shapes(overrides, code) -> None:
    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(_target_args(**overrides))

    assert captured.value.code == code


def test_preflight_rejects_shared_vector_multiperiod() -> None:
    scalar = _target_args()
    shared = MultiPeriodHPRTargetInputs(
        period_cases=[],
        selected_period_id="p0",
        selected_period_idx=0,
        hpr_type=scalar.hpr_type,
        max_multi_start=1,
        bb_minimiser=scalar.bb_minimiser,
    )

    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(shared)

    assert captured.value.code == "unsupported_multiperiod"


def test_preflight_rejects_wrong_input_type_and_backend() -> None:
    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(object())
    assert captured.value.code == "invalid_target_inputs"

    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(_target_args(simulation_backend="coolprop"))
    assert captured.value.code == "backend_mismatch"


def test_preflight_reports_missing_optional_dependency_before_fluid_work(
    monkeypatch,
) -> None:
    import OpenPinch.analysis.heat_pumps.performance_maps.targeting as targeting

    monkeypatch.setattr(targeting, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        targeting,
        "resolve_hpr_working_fluid",
        lambda *_args: pytest.fail("fluid resolution must not start"),
    )

    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(_target_args())

    assert captured.value.code == "dependency_unavailable"


@pytest.mark.parametrize(
    ("fluid", "kind", "component_count"),
    [
        ("R134a", "pure", 1),
        ("R407C", "registered_blend", 0),
        ("HEOS::R32[0.5]&R125[0.5]", "explicit_molar_mixture", 2),
        (
            "HEOS::R32[0.3]&R125[0.4]&R143a[0.3]",
            "explicit_molar_mixture",
            3,
        ),
    ],
)
def test_preflight_preserves_all_eligible_fluid_categories(
    fluid: str,
    kind: str,
    component_count: int,
) -> None:
    prepared = preflight_tespy_hpr_targeting(_target_args(refrigerant_ls=[fluid]))

    assert prepared.working_fluid.kind == kind
    assert len(prepared.working_fluid.components) == component_count


@pytest.mark.parametrize(
    ("fluid", "code"),
    [
        ("REFPROP::R134a", "property_backend_unsupported"),
        ("HEOS::R32[0.5]&invalid", "working_fluid_unsupported"),
        ("Air", "working_fluid_unsupported"),
    ],
)
def test_preflight_rejects_disallowed_or_unavailable_fluid_state(
    fluid: str,
    code: str,
) -> None:
    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(_target_args(refrigerant_ls=[fluid]))

    assert captured.value.code == code


def test_preflight_does_not_construct_a_tespy_network(monkeypatch) -> None:
    import OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy as adapter

    monkeypatch.setattr(
        adapter,
        "TespyHprPointSimulator",
        lambda: pytest.fail("preflight must not construct a TESPy session"),
    )

    preflight_tespy_hpr_targeting(_target_args())


def test_service_rejects_incompatible_tespy_before_optimizer_handler(
    monkeypatch,
) -> None:
    args = _target_args(n_cond=2)
    monkeypatch.setattr(hp, "construct_HPRTargetInputs", lambda **_kwargs: args)
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        args.hpr_type,
        lambda _args: pytest.fail("optimizer handler must not start"),
    )

    with pytest.raises(HprTargetCompatibilityError) as captured:
        hp._get_hpr_targets(
            Q_hpr_target=10.0,
            T_vals=np.array([100.0, 50.0]),
            H_hot=np.array([0.0, -10.0]),
            H_cold=np.array([10.0, 0.0]),
            config=object(),
            is_heat_pumping=True,
            simulation_backend="tespy",
        )

    assert captured.value.code == "unsupported_topology"


@seed(20260715)
@given(
    n_cond=st.integers(min_value=1, max_value=8),
    n_evap=st.integers(min_value=1, max_value=8),
)
def test_preflight_topology_property_accepts_only_one_by_one(
    n_cond: int,
    n_evap: int,
) -> None:
    args = _target_args(n_cond=n_cond, n_evap=n_evap)

    if (n_cond, n_evap) == (1, 1):
        assert preflight_tespy_hpr_targeting(args).cycle_id == (
            "single_stage_vapour_compression"
        )
    else:
        with pytest.raises(HprTargetCompatibilityError) as captured:
            preflight_tespy_hpr_targeting(args)
        assert captured.value.code == "unsupported_topology"


@seed(20260715)
@given(fluid=WORKING_FLUIDS, period_idx=st.integers(min_value=0, max_value=1000))
def test_preflight_fluid_and_scalar_period_property(
    fluid: str,
    period_idx: int,
) -> None:
    prepared = preflight_tespy_hpr_targeting(
        _target_args(refrigerant_ls=[fluid], period_idx=period_idx)
    )

    assert prepared.working_fluid.source_spec == fluid
    assert prepared.period_idx == period_idx
    if prepared.working_fluid.kind == "explicit_molar_mixture":
        assert sum(prepared.working_fluid.mole_fractions) == pytest.approx(1.0)


def test_preflight_rejects_bounds_without_positive_lift() -> None:
    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(
            _target_args(
                T_hot=np.array([120.0, 100.0]),
                T_cold=np.array([60.0, 50.0]),
            )
        )

    assert captured.value.code == "unsupported_state"


def test_preflight_defaults_empty_refrigerant_and_rejects_nonfinite_bounds() -> None:
    prepared = preflight_tespy_hpr_targeting(_target_args(refrigerant_ls=[]))
    assert prepared.working_fluid.source_spec.lower() == "water"

    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(
            _target_args(T_hot=np.array([np.nan]), T_cold=np.array([55.0]))
        )
    assert captured.value.code == "unsupported_state"


def test_preflight_translates_refrigerant_sort_failure(monkeypatch) -> None:
    import OpenPinch.analysis.heat_pumps.performance_maps.targeting as targeting

    monkeypatch.setattr(
        targeting, "PropsSI", lambda *_args: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HprTargetCompatibilityError) as captured:
        preflight_tespy_hpr_targeting(
            _target_args(refrigerant_ls=["R134a", "R32"], do_refrigerant_sort=True)
        )
    assert captured.value.code == "working_fluid_unsupported"
