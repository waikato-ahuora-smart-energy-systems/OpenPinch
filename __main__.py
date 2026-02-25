from pathlib import Path
from OpenPinch import PinchProblem

if __name__ == "__main__":
    from OpenPinch.examples.recreate_json import create_problem_and_results_json
    create_problem_and_results_json()

    # pp = PinchProblem()
    # pp.load(Path("OpenPinch/examples/OpenPinchWkbs/pulp_mill.xlsb"))
    # pp.target()
    # pp.export_to_Excel(Path("results"))
