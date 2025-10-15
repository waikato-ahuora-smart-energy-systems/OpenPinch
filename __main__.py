from pathlib import Path
from OpenPinch import PinchProblem

if __name__ == "__main__":
    pp = PinchProblem()
    pp.load(Path("examples/OpenPinchWkbs/pulp_mill.xlsb"))
    pp.target()
    pp.export_to_Excel(Path("results"))
