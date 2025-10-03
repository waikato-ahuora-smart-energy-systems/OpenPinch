import json
from pathlib import Path

import pytest

from OpenPinch import *
from OpenPinch.analysis.support_methods import get_value
from OpenPinch.lib import *


def get_example_problem_filepaths():
    test_data_dir = Path(__file__).resolve().parents[2] / "examples" / "stream_data"
    return [
        filepath
        for filepath in test_data_dir.iterdir()
        if filepath.name.startswith("p_") and filepath.name.endswith(".json")
    ]


def get_results_filepath(problem_filepath: Path) -> Path:
    """
    Given a problem file path (e.g. .../examples/stream_data/p_case.json),
    return the corresponding results file path (.../examples/results/r_case.json).
    Raise FileNotFoundError if the results file does not exist.
    """
    # Go up from stream_data → examples
    examples_dir = problem_filepath.parent.parent
    results_dir = examples_dir / "results"

    # Swap prefix p_ → r_
    result_name = problem_filepath.name.replace("p_", "r_", 1)
    result_path = results_dir / result_name

    # Validate existence
    if not result_path.exists():
        raise FileNotFoundError(
            f"Expected results file not found: {result_path} "
            f"(problem file was {problem_filepath})"
        )

    return result_path


@pytest.mark.parametrize("p_filepath", get_example_problem_filepaths())
def test_pinch_analysis_pipeline(p_filepath: Path):
    # Set the file path to the directory of this script
    with open(p_filepath) as json_data:
        data = json.load(json_data)

    r_filepath = get_results_filepath(p_filepath)

    project_name = p_filepath.stem[2:]

    res = pinch_analysis_service(data=data, project_name=project_name)

    # Get and validate the format of the "correct" targets from the Open Pinch workbook
    with open(r_filepath) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetOutput.model_validate(wkb_res)

    # Compare targets from Python and Excel implementations of Open Pinch
    if 1:
        for z in res.targets:
            for z0 in wkb_res.targets:
                if z.name in z0.name:
                    assert abs(get_value(z.Qh) - get_value(z0.Qh)) < 1e-3
                    assert abs(get_value(z.Qc) - get_value(z0.Qc)) < 1e-3
                    assert abs(get_value(z.Qr) - get_value(z0.Qr)) < 1e-3

                    for i in range(len(z.hot_utilities)):
                        assert (
                            abs(
                                get_value(z.hot_utilities[i].heat_flow)
                                - get_value(z0.hot_utilities[i].heat_flow)
                            )
                            < 1e-3
                        )

                    for i in range(len(z.cold_utilities)):
                        assert (
                            abs(
                                get_value(z.cold_utilities[i].heat_flow)
                                - get_value(z0.cold_utilities[i].heat_flow)
                            )
                            < 1e-3
                        )

    else:
        print(f"Name: {res.name}")
        for z in res.targets:
            for z0 in wkb_res.targets:
                if z.name in z0.name:
                    print("")
                    print("Name:", z.name, z0.name)
                    print(
                        "Qh:",
                        round(get_value(z.Qh), 2),
                        "Qh:",
                        round(get_value(z0.Qh), 2),
                        round(get_value(z.Qh), 2) == round(get_value(z0.Qh), 2),
                        sep="\t",
                    )
                    print(
                        "Qc:",
                        round(get_value(z.Qc), 2),
                        "Qc:",
                        round(get_value(z0.Qc), 2),
                        round(get_value(z.Qc), 2) == round(get_value(z0.Qc), 2),
                        sep="\t",
                    )
                    print(
                        "Qr:",
                        round(get_value(z.Qr), 2),
                        "Qr:",
                        round(get_value(z0.Qr), 2),
                        round(get_value(z.Qr), 2) == round(get_value(z0.Qr), 2),
                        sep="\t",
                    )
                    [
                        print(
                            z.hot_utilities[i].name + ":",
                            round(get_value(z.hot_utilities[i].heat_flow), 2),
                            z0.hot_utilities[i].name + ":",
                            round(get_value(z0.hot_utilities[i].heat_flow), 2),
                            round(get_value(z.hot_utilities[i].heat_flow), 2)
                            == round(get_value(z0.hot_utilities[i].heat_flow), 2),
                            sep="\t",
                        )
                        for i in range(len(z.hot_utilities))
                    ]
                    [
                        print(
                            z.cold_utilities[i].name + ":",
                            round(get_value(z.cold_utilities[i].heat_flow), 2),
                            z0.cold_utilities[i].name + ":",
                            round(get_value(z0.cold_utilities[i].heat_flow), 2),
                            round(get_value(z.cold_utilities[i].heat_flow), 2)
                            == round(get_value(z0.cold_utilities[i].heat_flow), 2),
                            sep="\t",
                        )
                        for i in range(len(z.cold_utilities))
                    ]
                    print("")
