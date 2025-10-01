from OpenPinch import PinchProblem
from pathlib import Path

if __name__ == "__main__":
    P = PinchProblem(
        problem_filepath = Path(__file__).resolve().parents[0] / "Excel_Version" / "Data_input_template.xlsx",
        results_dir = Path(__file__).resolve().parents[0] / "results",
    )
