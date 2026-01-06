from pathlib import Path
from OpenPinch import PinchProblem

if __name__ == "__main__":
    pp = PinchProblem()
    pp.load(Path("OpenPinch/examples/OpenPinchWkbs/UnderReview/Meat Processing Factory.xlsb"))
    pp.target()
    pp.export_to_Excel(Path("results"))
