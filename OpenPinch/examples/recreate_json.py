import os
from OpenPinch.utils import *

def create_problem_and_results_json():
    # Set the file path to the directory of this script
    filepath_load = os.path.dirname(__file__)
    filepath_save = os.path.dirname(__file__)

    for filename in os.listdir(filepath_load):
        if (filename.endswith(".xlsb") or filename.endswith(".xlsx")) and not filename.startswith("~$"):
            excel_file = os.path.join(filepath_load, filename)
            project_name = os.path.splitext(filename)[0]
            p_json_file = filepath_save + "/p_" + project_name + ".json"
            get_problem_from_excel(excel_file, p_json_file)

            r_json_file = filepath_save + "/r_" + project_name + ".json"
            get_results_from_excel(excel_file, r_json_file, project_name)
