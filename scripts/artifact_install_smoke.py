"""Smoke-check an installed OpenPinch wheel without importing the checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Checkout path that the installed package must not resolve from.",
    )
    parser.add_argument(
        "--surface",
        choices=("core", "tespy"),
        default="core",
        help="Installed dependency surface to exercise.",
    )
    return parser


def is_checkout_source_import(package_path: Path, repo_root: Path) -> bool:
    """Return whether ``package_path`` resolves from the checkout source package."""
    source_package = repo_root.resolve() / "OpenPinch"
    return package_path.resolve().is_relative_to(source_package)


def _exercise_tespy_public_target_and_map() -> None:
    """Run the bounded public target-to-map path from the installed package."""
    import numpy as np

    import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as cascade
    from OpenPinch import PinchProblem
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
        translate_hpr_result,
    )
    from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest
    from OpenPinch.resources import read_sample_case

    def bounded_candidate_search(*, f_obj, x0_ls, bnds, args):
        point = np.array([(lower + upper) / 2.0 for lower, upper in bnds])
        point[0] = 0.0
        point[1] = 0.35
        point[2] = 0.20
        point[3] = 0.02
        point[4] = 0.50
        point[5] = 1.0
        point[-1] = 0.0
        result = f_obj(point, args)
        if not result.success:
            raise AssertionError(
                f"Installed TESPy target candidate failed: {result.failure_reason}"
            )
        return translate_hpr_result(result, ambient_args=args)

    original_solve = cascade.solve_hpr_placement
    cascade.solve_hpr_placement = bounded_candidate_search
    try:
        problem = PinchProblem(
            json.loads(read_sample_case("pulp_mill.json")),
            project_name="Installed TESPy HPR smoke",
        )
        target = problem.target.vapour_compression_heat_pump(
            simulation_backend="tespy",
            condensers=1,
            evaporators=1,
            refrigerants=["ammonia"],
            initialize_from_carnot=False,
            allow_integrated_expander=False,
            load_fraction=0.25,
            maximum_restarts=1,
        )
    finally:
        cascade.solve_hpr_placement = original_solve

    record = target.hpr_details.target_simulation_record
    if record is None or record.simulation_backend != "tespy":
        raise AssertionError("Installed TESPy target is missing its winning record.")
    performance_map = problem.target.hpr_performance_map(
        target=target,
        request=HprPerformanceMapRequest(
            map_id="installed-tespy-hpr-smoke",
            source_temperatures=[
                record.nominal_evaporating_temperature
                + record.source_approach_temperature
            ],
            sink_temperatures=[
                record.nominal_condensing_temperature - record.sink_approach_temperature
            ],
            load_fractions=[0.75, 1.0],
        ),
    )
    if (
        performance_map.thermodynamic_backend != "tespy"
        or len(performance_map.points) != 2
    ):
        raise AssertionError("Installed wheel failed its public TESPy HPR map.")


def main(argv: list[str] | None = None) -> int:
    """Exercise the installed package, CLI, and packaged resources."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    import OpenPinch
    from OpenPinch import PinchProblem, PinchWorkspace
    from OpenPinch.analysis.heat_pumps.performance_maps import HprTargetMapBasis
    from OpenPinch.analysis.heat_pumps.performance_maps.resources import (
        load_tespy_compressor_characteristic,
        read_tespy_compressor_characteristic_bytes,
    )
    from OpenPinch.contracts.hpr import HprTargetSimulationRecord
    from OpenPinch.contracts.hpr_performance_map import HprPerformanceMap
    from OpenPinch.resources import (
        list_hpr_performance_map_contract_resources,
        list_notebooks,
        list_sample_cases,
        load_hpr_performance_map_contract_resource,
        read_sample_case,
    )

    package_path = Path(OpenPinch.__file__).resolve()
    if is_checkout_source_import(package_path, repo_root):
        raise AssertionError(f"Imported OpenPinch from checkout: {package_path}")

    characteristic = load_tespy_compressor_characteristic()
    if (
        HprTargetMapBasis is None
        or characteristic.characteristic_set_id
        != "openpinch-single-stage-compressor-v1"
        or len(read_tespy_compressor_characteristic_bytes()) != 458
    ):
        raise AssertionError("Installed wheel contains an invalid TESPy model asset.")
    if args.surface == "tespy":
        from OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy import (
            TespyHprPointSimulator,
        )
        from OpenPinch.analysis.heat_pumps.performance_maps.factory import (
            get_hpr_point_simulator,
        )

        if not isinstance(get_hpr_point_simulator("tespy"), TespyHprPointSimulator):
            raise AssertionError("Installed wheel could not create a TESPy simulator.")
        _exercise_tespy_public_target_and_map()
    else:
        try:
            __import__("tespy")
        except ImportError:
            pass
        else:
            raise AssertionError("Core artifact smoke unexpectedly installed TESPy.")

    notebooks = list_notebooks()
    sample_cases = list_sample_cases()
    if not notebooks or not sample_cases:
        raise AssertionError("Installed wheel is missing packaged resources.")
    sample = read_sample_case("basic_pinch.json")
    if not sample:
        raise AssertionError("Installed wheel could not read basic_pinch.json.")
    if "process_mvr.json" not in sample_cases:
        raise AssertionError("Installed wheel is missing process_mvr.json.")
    if "11_process_mvr_and_cascade.ipynb" not in notebooks:
        raise AssertionError("Installed wheel is missing the Process MVR tutorial.")

    hpr_resources = list_hpr_performance_map_contract_resources()
    expected_hpr_resources = [
        "heat-pump-1.0.json",
        "refrigeration-1.0.json",
        "schema-1.0.json",
    ]
    if hpr_resources != expected_hpr_resources:
        raise AssertionError(f"Unexpected HPR contract resources: {hpr_resources}")
    schema = load_hpr_performance_map_contract_resource("schema-1.0.json")
    if schema.get("title") != "HprPerformanceMap" or "points" not in schema.get(
        "properties", {}
    ):
        raise AssertionError("Installed wheel contains an invalid HPR map schema.")
    for fixture_name in expected_hpr_resources[:2]:
        payload = load_hpr_performance_map_contract_resource(fixture_name)
        performance_map = HprPerformanceMap.model_validate(payload)
        if performance_map.schema_version != "1.0" or not performance_map.points:
            raise AssertionError(f"Installed wheel contains invalid {fixture_name}.")
    if HprTargetSimulationRecord.model_json_schema()["title"] != (
        "HprTargetSimulationRecord"
    ):
        raise AssertionError("Installed wheel is missing the HPR target record.")

    problem = PinchProblem(json.loads(sample), project_name="Wheel contract")
    direct = problem.target.direct_heat_integration()
    if problem.results.name != "Wheel contract" or not problem.results.targets:
        raise AssertionError("Installed wheel failed the PinchProblem workflow.")
    inverse = problem.target.heat_recovery_dt_min(
        heat_recovery=float(direct.heat_recovery_target)
    )
    if abs(inverse.dt_min.value - 10.0) > 2e-6:
        raise AssertionError("Installed wheel failed inverse heat-recovery targeting.")
    workspace = PinchWorkspace(json.loads(sample), project_name="Wheel contract")
    if workspace.list_cases() != ["baseline"]:
        raise AssertionError("Installed wheel failed the PinchWorkspace workflow.")

    if OpenPinch.__all__ != ["PinchProblem", "PinchWorkspace"]:
        raise AssertionError(f"Unexpected root exports: {OpenPinch.__all__}")
    forbidden_root_exports = {
        "HprPerformanceMap",
        "HprPerformanceMapRequest",
        "HprTargetSimulationRecord",
        "TargetInput",
        "TargetOutput",
    }
    leaked = sorted(name for name in forbidden_root_exports if hasattr(OpenPinch, name))
    if leaked:
        raise AssertionError(f"Root package exposes unexpected aliases: {leaked}")

    retired_packages = (
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.services",
        "OpenPinch.streamlit_webviewer",
        "OpenPinch.utils",
    )
    resolved = sorted(
        package
        for package in retired_packages
        if importlib.util.find_spec(package) is not None
    )
    if resolved:
        raise AssertionError(f"Installed wheel contains retired packages: {resolved}")

    subprocess.run(
        [sys.executable, "-m", "OpenPinch", "notebook", "--help"],
        check=True,
        cwd=repo_root.parent,
    )
    print(f"Installed artifact smoke passed for {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
