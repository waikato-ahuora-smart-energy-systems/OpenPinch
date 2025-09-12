import os
from OpenPinch.utils import get_problem_from_excel, get_results_from_excel


# def recreate_full_example_from_wkbs():
#     # Set the file path to the directory of this script
#     filepath_load = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + 'OpenPinch/tests/OpenPinchWkbs'
#     filepath_save = os.path.dirname(os.path.dirname(__file__)) + "/test_OpenPinch/test_data"

#     for filename in os.listdir(filepath_load):
#         # filename = 'locally_integrated.xlsb'
#         if (filename.endswith(".xlsb") or filename.endswith(".xlsx")) and not filename.startswith("~$"):
#             excel_file = os.path.join(filepath_load, filename)
#             project_name = os.path.splitext(filename)[0]
#             p_json_file = filepath_save + "/p_" + project_name + ".json"
#             get_problem_from_excel(excel_file, p_json_file)
            
#             r_json_file = filepath_save + "/r_" + project_name + ".json"
#             get_results_from_excel(excel_file, r_json_file, project_name)
