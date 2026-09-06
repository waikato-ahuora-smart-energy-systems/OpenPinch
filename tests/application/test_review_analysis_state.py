"""PR 96 regressions for scope and backend transitions in prepared analysis."""

from copy import deepcopy
from unittest.mock import patch

import pytest
from hypothesis import example, given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import DirectHeatPumpTarget, DirectRefrigerationTarget
from tests.application.test_hpr_period_batch_boundaries import _two_period_payload


def _sibling_problem():
    payload = _two_period_payload()
    payload["zone_tree"]["children"].append({"name": "AreaB", "type": "Process Zone"})
    for stream in deepcopy(payload["streams"]):
        stream["zone"] = "Site/AreaB"
        stream["name"] = stream["name"].replace("A", "B")
        payload["streams"].append(stream)
    return PinchProblem(payload, project_name="Site")


def _assert_scope(outputs, zone):
    assert list(outputs) == ["base", "peak"]
    for sid, output in outputs.items():
        assert output.targets
        assert {row.scope for row in output.targets} == {f"Site/{zone}"}
        assert {row.period_id for row in output.targets} == {sid}
        assert output.graphs
        assert {graph.zone_address for graph in output.graphs.values()} == {
            f"Site/{zone}"
        }


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("include_subzones", [False, True])
def test_period_batches_exclude_previously_selected_sibling(workers, include_subzones):
    problem = _sibling_problem()
    problem.target.all_periods.direct_heat_integration(zone="AreaA", workers=workers)
    outputs = problem.target.all_periods.direct_heat_integration(
        zone="AreaB", workers=workers, include_subzones=include_subzones
    )
    _assert_scope(outputs, "AreaB")
    _assert_scope(problem.period_results, "AreaB")


@seed(20260715)
@settings(max_examples=8, deadline=None)
@given(
    zones=st.lists(st.sampled_from(["AreaA", "AreaB"]), min_size=1, max_size=4),
    workers=st.sampled_from([1, 2]),
)
def test_generated_scope_transitions_keep_enrichment_local(zones, workers):
    problem = _sibling_problem()
    for zone in zones:
        thermal = problem.target.all_periods.direct_heat_integration(
            zone=zone, workers=workers
        )
        _assert_scope(thermal, zone)
        with patch(
            "OpenPinch.application.targeting.compute_direct_integration_targets",
            side_effect=AssertionError("Compatible thermal state must be reused"),
        ):
            enriched = problem.target.all_periods.exergy(zone=zone, workers=workers)
        _assert_scope(enriched, zone)
        for sid, output in enriched.items():
            assert output.targets[0].exergy_sources is not None
            assert output.targets[0].Qr == thermal[sid].targets[0].Qr


def _controlled_hpr(zone, args, *, refrigeration=False):
    """Replace only the scientific solver; exercise real application lifecycle."""
    direct = zone.targets["Direct Integration"]
    family = DirectRefrigerationTarget if refrigeration else DirectHeatPumpTarget
    target = family(
        scope=zone.address,
        type="Direct Refrigeration" if refrigeration else "Direct Heat Pump",
        period_id=direct.period_id,
        period_idx=direct.period_idx,
        config=deepcopy(zone.config),
        parent_zone=zone.parent_zone,
        pt=deepcopy(direct.pt),
        hpr_success=True,
        hpr_simulation_backend=args["simulation_backend"],
        hpr_cycle="Cascade vapour compression cycles",
        hpr_utility_total=0.0,
        hpr_work=0.0,
        hpr_external_utility=0.0,
        hpr_ambient_hot=0.0,
        hpr_ambient_cold=0.0,
        hpr_cop=4.0,
        hpr_eta_he=0.0,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details=None,
    )
    zone.add_target(target)
    return zone


@pytest.mark.parametrize("refrigeration", [False, True])
@seed(20260715)
@settings(max_examples=8, deadline=None)
@example(backends=["coolprop", "tespy", "coolprop"])
@given(
    backends=st.lists(st.sampled_from(["coolprop", "tespy"]), min_size=2, max_size=5)
)
def test_backend_transitions_change_identity_and_reject_old_references(
    refrigeration, backends
):
    from OpenPinch.application._problem.targeting.provenance import validate_base_target

    problem = _sibling_problem()
    problem.target.direct_heat_integration(zone="AreaA", period_id="base")
    name = "refrigeration" if refrigeration else "heat_pump"
    method = getattr(problem.target, f"vapour_compression_{name}")
    previous = None
    with patch(
        f"OpenPinch.application._problem.accessors.target.direct_{name}_service",
        side_effect=lambda zone, args: _controlled_hpr(
            zone, args, refrigeration=refrigeration
        ),
    ):
        for backend in backends:
            target = method(
                zone="AreaA", period_id="base", simulation_backend=backend.upper()
            )
            assert target.provenance.effective_settings["simulation_backend"] == backend
            if previous is not None:
                changed = previous.hpr_simulation_backend != backend
                assert (previous.provenance != target.provenance) == changed
                if changed:
                    before = problem.results.model_dump(mode="json")
                    with pytest.raises(ValueError, match="base_target"):
                        validate_base_target(
                            problem, previous, zone="AreaA", period_id="base"
                        )
                    if not refrigeration:
                        with pytest.raises(ValueError, match="base_target"):
                            problem.target.exergy(
                                zone="AreaA", period_id="base", base_target=previous
                            )
                    assert problem.results.model_dump(mode="json") == before
            previous = target
