# OpenPinch

OpenPinch is an open-source toolkit for advanced Pinch Analysis and Total Site
Integration. 

## History

OpenPinch started in 2011 as Excel Workbook with macros. Since its inception, the workbook was developed in multiple directions, including Total Site Heat Integration, multiple utility targeting, retrofit targeting, cogeneration targeting, and more. The latest version of the Excel Workbook is free-to-use and available in the "Excel Version" folder on the OpenPinch github repository. 

In 2021, a Python implementation of OpenPinch began, bringing the capabilities of the long-running Excel workbook into a modern Python API. The goal is to provide a sound basis for research, development and application. It is also freely available for integrating with other software tools and projects, and embeding results 
into wider optimisation workflows.

## Citation

In scientific works, please cite this github repository, including the Pypi version number. Forks of OpenPinch should also reference back to this source ideally.

At present, a publication for citation is under peer-review, and the approperiate reference will be provided in due course.

## Highlights

- Multi-scale targeting: unit operation, process, site, community, and regional zones
- Direct heat integration targeting and indirect heat integration targeting (via the utility system)
- Multiple utility targeting (isothermal and non-isothermal)
- Grand composite curve (GCC) manipulation and visualisation helpers
- Excel template for importing data 
- Visualisation via a Streamlit web application
- Pydantic schema models for validated programmatic workflows

## Installation

Install the latest published release from PyPI:

```bash
python -m pip install openpinch
```

## Quickstart

The high-level service accepts Excel data input via the template format. Copy and edit the Excel template (identical to the OpenPinch Excel Workbook) to input stream and utility data. 

```python
from pathlib import Path
from OpenPinch import PinchProblem

pp = PinchProblem()
pp.load(Path("[location]/[filname].xlsb"))
pp.target()
pp.export_to_Excel(Path("results"))
```

Alteratively, one can define each individual stream following the defined schema. 

```python
from OpenPinch import PinchProblem
from OpenPinch.lib.schema import TargetInput, StreamSchema, UtilitySchema
from OpenPinch.lib.enums import StreamType

streams = [
    StreamSchema(
        zone="Process Unit",
        name="Reboiler Vapor",
        t_supply=200.0,
        t_target=120.0,
        heat_flow=8000.0,
        dt_cont=10.0,
        htc=1.5,
    ),
    StreamSchema(
        zone="Process Unit",
        name="Feed Preheat",
        t_supply=40.0,
        t_target=160.0,
        heat_flow=6000.0,
        dt_cont=10.0,
        htc=1.2,
    ),
]

utilities = [
    UtilitySchema(
        name="Cooling Water",
        type=StreamType.Cold,
        t_supply=25.0,
        t_target=35.0,
        heat_flow=120000.0,
        dt_cont=5.0,
        htc=0.8,
        price=12.0,
    )
]

input_data = TargetInput(streams=streams, utilities=utilities)

pp = PinchProblem()
pp.load(input_data)
pp.target()
pp.export_to_Excel(Path("results"))
```

## Documentation

Full documentation (getting started, guides, and API reference) is available:

https://openpinch.readthedocs.io/en/latest/

Please note: the reference guide, like the repository, is under development. Errors are likely due the research nature of the project. 

## Testing

Install the project in editable mode along with any optional test dependencies,
then run the test suite with:

```bash
python -m pip install -e .
pytest
```

## Contributors

Founder: Dr Tim Walmsley, University of Waikato


Stephen Burroughs, Benjamin Lincoln, Alex Geary, Harrison Whiting, Khang Tran, Roger Padullés, Jasper Walden

## Contributing

Issues and pull requests are welcome! Please open a discussion if you have questions about data formats or feature ideas. When submitting code, aim for:

- Typed interfaces and clear docstring
- Small methods with singular purpose
- Pytests covering new behaviour
- Updated documentation where relevant

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
