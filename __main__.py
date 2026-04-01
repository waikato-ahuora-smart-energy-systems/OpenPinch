"""Local entry point for running a bundled OpenPinch example case."""

from pathlib import Path

from OpenPinch import *

if __name__ == "__main__":
    problem_path = Path("enter path to the source problem file(s)")
    export_path = Path("enter path to the export folder")
    
    pp = PinchProblem()
    pp.load(problem_path)
    pp.target()
    pp.export_to_Excel(export_path)
