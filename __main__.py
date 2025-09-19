# from OpenPinch import PinchProblem
import OpenPinch as op
from pathlib import Path


if __name__ == "__main__":
    P = op.PinchProblem(
        project_name = "",
        problem_filepath = Path(__file__).resolve().parents[0] / "Excel_Version" / "Data_input_template.xlsx",
        results_dir = Path(__file__).resolve().parents[0] / "results",
    )
       