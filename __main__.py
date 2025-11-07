from pathlib import Path
from OpenPinch import PinchProblem

if __name__ == "__main__":
    pp = PinchProblem()
    pp.load(Path("examples/OpenPinchWkbs/potatoe-simple.xlsb"))
    pp.target()
    pp.export_to_Excel(Path("results"))
