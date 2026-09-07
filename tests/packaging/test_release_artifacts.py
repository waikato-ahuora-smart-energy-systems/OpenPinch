"""Release artifact checks for packaging boundaries."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

from hypothesis import given, seed

from scripts.artifact_install_smoke import is_checkout_source_import
from tests.strategies.artifact_paths import artifact_import_path_cases
from tests.support.paths import REPOSITORY_ROOT

HPR_CONTRACT_RESOURCE_ROOT = Path("OpenPinch/contracts/resources/hpr_performance_map")
HPR_CONTRACT_RESOURCES = (
    "heat-pump-1.0.json",
    "refrigeration-1.0.json",
    "schema-1.0.json",
)
TESPY_CHARACTERISTIC_RESOURCE = Path(
    "OpenPinch/analysis/heat_pumps/performance_maps/characteristics/"
    "openpinch-single-stage-compressor-v1.json"
)


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path, list[str], list[str]]:
    repo_root = REPOSITORY_ROOT
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_dist.py",
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    wheel_paths = sorted(out_dir.glob("*.whl"))
    sdist_paths = sorted(out_dir.glob("*.tar.gz"))
    assert wheel_paths, (
        f"no wheel created in {out_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert sdist_paths, (
        f"no sdist created in {out_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    wheel_path = wheel_paths[0]
    sdist_path = sdist_paths[0]

    with ZipFile(wheel_path) as wheel:
        wheel_names = wheel.namelist()

    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_names = sdist.getnames()

    return wheel_path, sdist_path, wheel_names, sdist_names


def _assert_common_release_boundary(names: list[str], *, root_prefix: str = "") -> None:
    prefix = f"{root_prefix}/" if root_prefix else ""

    assert f"{prefix}OpenPinch/__init__.py" in names
    assert f"{prefix}OpenPinch/main.py" not in names
    assert f"{prefix}__init__.py" not in names

    required_owner_files = [
        "OpenPinch/domain/stream.py",
        "OpenPinch/contracts/hpr.py",
        "OpenPinch/contracts/hpr_performance_map.py",
        "OpenPinch/contracts/input.py",
        "OpenPinch/optimisation/service.py",
        "OpenPinch/application/problem.py",
        "OpenPinch/analysis/targeting/cascade.py",
        "OpenPinch/analysis/heat_pumps/performance_maps/target_basis.py",
        "OpenPinch/analysis/heat_pumps/performance_maps/target_records.py",
        "OpenPinch/analysis/heat_pumps/performance_maps/targeting.py",
        "OpenPinch/analysis/heat_exchanger_networks/results/assembly.py",
        "OpenPinch/adapters/io/json.py",
        "OpenPinch/presentation/reporting/results.py",
    ]
    for owner_file in required_owner_files:
        assert f"{prefix}{owner_file}" in names
    for resource_name in HPR_CONTRACT_RESOURCES:
        resource_path = HPR_CONTRACT_RESOURCE_ROOT / resource_name
        assert f"{prefix}{resource_path.as_posix()}" in names
    assert f"{prefix}{TESPY_CHARACTERISTIC_RESOURCE.as_posix()}" in names

    forbidden_prefixes = [
        f"{prefix}examples/",
        f"{prefix}tests/",
        f"{prefix}Excel_Version/",
        f"{prefix}OpenPinch/classes/",
        f"{prefix}OpenPinch/lib/",
        f"{prefix}OpenPinch/services/",
        f"{prefix}OpenPinch/utils/",
        f"{prefix}OpenPinch/streamlit_webviewer/",
    ]
    forbidden_fragments = [
        "streamlit_app.py",
        "OpenPinch/adapters/io/target_workbook.py",
        "OpenPinch/presentation/dashboard/problem.py",
        "OpenPinch/presentation/reporting/targets.py",
    ]

    for disallowed_prefix in forbidden_prefixes:
        assert not any(name.startswith(disallowed_prefix) for name in names)
    for fragment in forbidden_fragments:
        assert not any(fragment in name for name in names)

    assert any(
        name.endswith("OpenPinch/tutorials/sample_cases/basic_pinch.json")
        for name in names
    )
    assert any(
        name.endswith("OpenPinch/tutorials/sample_cases/heat_pump_targeting.json")
        for name in names
    )
    assert any(
        name.endswith(
            "OpenPinch/tutorials/sample_cases/Four-stream-Yee-and-Grossmann-1990-1.json"
        )
        for name in names
    )
    assert any(
        name.endswith(
            "OpenPinch/tutorials/notebooks/01_first_solve_and_core_curves.ipynb"
        )
        for name in names
    )
    assert any(
        name.endswith("OpenPinch/tutorials/sample_cases/process_mvr.json")
        for name in names
    )
    assert any(
        name.endswith("OpenPinch/tutorials/notebooks/11_process_mvr_and_cascade.ipynb")
        for name in names
    )


def test_release_artifacts_exclude_repo_only_assets(tmp_path: Path):
    wheel_path, sdist_path, wheel_names, sdist_names = _build_artifacts(tmp_path)
    _assert_common_release_boundary(wheel_names)

    sdist_root = next(name.split("/", 1)[0] for name in sdist_names if "/" in name)
    _assert_common_release_boundary(sdist_names, root_prefix=sdist_root)

    with ZipFile(wheel_path) as wheel, tarfile.open(sdist_path, "r:gz") as sdist:
        for resource_name in HPR_CONTRACT_RESOURCES:
            relative_path = HPR_CONTRACT_RESOURCE_ROOT / resource_name
            source_bytes = (REPOSITORY_ROOT / relative_path).read_bytes()
            wheel_bytes = wheel.read(relative_path.as_posix())
            member = sdist.extractfile(f"{sdist_root}/{relative_path.as_posix()}")

            assert member is not None
            assert wheel_bytes == source_bytes == member.read()
            maximum_size = (
                250 * 1024 if resource_name.startswith("schema-") else 100 * 1024
            )
            assert len(source_bytes) < maximum_size

        characteristic_source = (
            REPOSITORY_ROOT / TESPY_CHARACTERISTIC_RESOURCE
        ).read_bytes()
        characteristic_wheel = wheel.read(TESPY_CHARACTERISTIC_RESOURCE.as_posix())
        characteristic_member = sdist.extractfile(
            f"{sdist_root}/{TESPY_CHARACTERISTIC_RESOURCE.as_posix()}"
        )
        assert characteristic_member is not None
        assert characteristic_wheel == characteristic_source
        assert characteristic_member.read() == characteristic_source
        assert len(characteristic_source) == 458


def test_checkout_local_virtual_environment_is_not_source_import(tmp_path: Path):
    repo_root = tmp_path / "OpenPinch"
    source_path = repo_root / "OpenPinch" / "__init__.py"
    installed_path = (
        repo_root
        / ".venv"
        / "lib"
        / "python3.14"
        / "site-packages"
        / "OpenPinch"
        / "__init__.py"
    )

    assert is_checkout_source_import(source_path, repo_root)
    assert not is_checkout_source_import(installed_path, repo_root)


@seed(20260715)
@given(case=artifact_import_path_cases())
def test_only_source_package_imports_are_classified_as_checkout_source(case):
    assert (
        is_checkout_source_import(case.package_path, case.repo_root)
        is case.source_import
    )
