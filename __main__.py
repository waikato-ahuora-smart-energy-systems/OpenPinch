from OpenPinch import main
from pathlib import Path
import json

def run_from_json():
    p_file_path = Path(__file__).resolve().parents[0] / "examples" / "stream_data" / "p_illustrative.json"
    r_file_path = Path(__file__).resolve().parents[0] / "examples" / "results" / "r_illustrative.json"

    with open(p_file_path) as f:
        input_data = json.load(f)

    res = main.targeting_analysis_from_pinch_service(input_data)
    

    with open(r_file_path) as f:
        wkb_res = json.load(f)
    wkb_res = main.TargetResponse.model_validate(wkb_res)    

    return res 


def run_from_excel():
    file_path = Path(__file__).resolve().parents[0] / "Excel_Version" / "Data_input_template.xlsx"
    input_data = main.get_problem_from_excel(file_path)
    res = main.targeting_analysis_from_pinch_service(input_data)
    return res


if __name__ == "__main__":
    # res = run_from_json()
    res = run_from_excel()
    print(res)
       