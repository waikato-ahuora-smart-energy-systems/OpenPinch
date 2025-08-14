import pytest
import json, os
from src.lib import *
from src.main import *
from src.analysis.support_methods import get_value


def get_test_filenames():
    test_data_dir = os.path.dirname(__file__) + '/test_data'
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]

@pytest.mark.parametrize("filename", get_test_filenames())
def test_pinch_analysis_pipeline(filename):

    # Set the file path to the directory of this script
    filepath = os.path.dirname(__file__) + '/test_data'
    p_file_path = filepath + '/p_' + filename[2:]
    r_file_path = filepath + '/r_' + filename[2:]
    with open(p_file_path) as json_data:
        data = json.load(json_data)

    project_name = filename[2:-5]

    # Validate request data using Pydantic model
    request_data = TargetRequest.model_validate(data)

    # Perform advanced pinch analysis and total site analysis
    return_data = target(
        zone_tree=request_data.zone_tree,
        streams=request_data.streams,
        utilities=request_data.utilities,
        options=request_data.options,
        name=project_name,
    )

    # Validate response data
    res = TargetResponse.model_validate(return_data)

    # Get and calidate the format of the "correct" targets from the Open Pinch workbook
    with open(r_file_path) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetResponse.model_validate(wkb_res)

    # Compare targets from Python and Excel implementations of Open Pinch
    if 1:
        for z in res.targets:
            for z0 in wkb_res.targets:
                if z.name in z0.name:
                    assert abs(get_value(z.Qh) - get_value(z0.Qh)) < 1e-3
                    assert abs(get_value(z.Qc) - get_value(z0.Qc)) < 1e-3
                    assert abs(get_value(z.Qr) - get_value(z0.Qr)) < 1e-3

                    for i in range(len(z.hot_utilities)):
                        assert abs(get_value(z.hot_utilities[i].heat_flow) - get_value(z0.hot_utilities[i].heat_flow)) < 1e-3

                    for i in range(len(z.cold_utilities)):
                        assert abs(get_value(z.cold_utilities[i].heat_flow) - get_value(z0.cold_utilities[i].heat_flow)) < 1e-3

    else:
        print(f'Name: {res.name}')
        for z in res.targets:
            for z0 in wkb_res.targets:
                if z.name in z0.name:
                    print('')
                    print('Name:', z.name, z0.name)
                    print('Qh:', round(get_value(z.Qh), 2), 'Qh:', round(get_value(z0.Qh), 2), round(get_value(z.Qh), 2)==round(get_value(z0.Qh), 2), sep='\t')
                    print('Qc:', round(get_value(z.Qc), 2), 'Qc:', round(get_value(z0.Qc), 2), round(get_value(z.Qc), 2)==round(get_value(z0.Qc), 2), sep='\t')
                    print('Qr:', round(get_value(z.Qr), 2), 'Qr:', round(get_value(z0.Qr), 2), round(get_value(z.Qr), 2)==round(get_value(z0.Qr), 2), sep='\t')
                    [print(z.hot_utilities[i].name + ':', round(get_value(z.hot_utilities[i].heat_flow), 2), z0.hot_utilities[i].name + ':', round(get_value(z0.hot_utilities[i].heat_flow), 2), round(get_value(z.hot_utilities[i].heat_flow), 2)==round(get_value(z0.hot_utilities[i].heat_flow), 2), sep='\t') for i in range(len(z.hot_utilities))]
                    [print(z.cold_utilities[i].name + ':', round(get_value(z.cold_utilities[i].heat_flow), 2), z0.cold_utilities[i].name + ':', round(get_value(z0.cold_utilities[i].heat_flow), 2), round(get_value(z.cold_utilities[i].heat_flow), 2)==round(get_value(z0.cold_utilities[i].heat_flow), 2), sep='\t') for i in range(len(z.cold_utilities))]
                    print('')
