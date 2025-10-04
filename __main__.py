# from pathlib import Path

# from OpenPinch import PinchProblem

# if __name__ == "__main__":
#     P = PinchProblem(
#         problem_filepath = Path(__file__).resolve().parents[0] / "Excel_Version" / "Data_input_template.xlsx",
#         results_dir = Path(__file__).resolve().parents[0] / "results",
#     )

from pathlib import Path
from OpenPinch import PinchProblem

problem = PinchProblem(
    problem_filepath=Path("examples/OpenPinchWkbs/pulp_mill.xlsb"),
    results_dir=Path("results"),
    run=True,
)
