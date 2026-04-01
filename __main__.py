"""Local entry point for running a bundled OpenPinch example case."""

from pathlib import Path

from OpenPinch import *

if __name__ == "__main__":
    # from OpenPinch.examples.recreate_json import create_problem_and_results_json
    # create_problem_and_results_json()
    
    pp = PinchProblem()
    pp.load(Path("Hidden/ibericht_amrein_futtermuehle_hslu_2014.xlsx"))
    pp.target()
    pp.export_to_Excel(Path("results"))
