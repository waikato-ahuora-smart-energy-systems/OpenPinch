# OpenPinch

OpenPinch is an open-source toolkit for advanced Pinch Analysis and total site
integration. 

## History

OpenPinch started as part of Tim Walmsley's PhD research at the University of
Waikato and has continued evolving through industrial projects and community
contributions. This implementation brings the capabilities of the long-running 
Excel/VBA workbook into a modern Python API so engineers can automate targeting 
studies, integrate with other software tools and projects, and embed results 
into wider optimisation workflows.

At present, a publication for citation is under preparation, and the approperiate
reference will be provided in due course. 

## Highlights

- Multi-scale targeting for process, site, and regional studies
- Multiple utility targeting (isothermal and non-isothermal) with assisted heat
  integration options
- Grand composite curve (GCC) manipulation and visualisation helpers
- Imports the established Excel data templates and exports detailed reports
- Pydantic schema models for validated programmatic workflows

## Installation

Install the latest published release from PyPI:

```bash
python -m pip install openpinch
```

For local development, clone the repository and install it in editable mode
along with the documentation requirements:

```bash
git clone https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch.git
cd OpenPinch
python -m pip install -e .
python -m pip install -r docs/requirements.txt
```

## Quickstart

The high-level service accepts raw payloads (dicts, Pydantic models, etc.) and
returns validated targeting results:

```python
from OpenPinch import pinch_analysis_service
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

payload = TargetInput(streams=streams, utilities=utilities)
results = pinch_analysis_service(payload, project_name="Demo Site")

for target in results.targets:
    print(target.name, target.Qh, target.Qc)
```

Prefer to orchestrate loading and exporting via files? Wrap the workflow with
`PinchProblem`:

```python
from OpenPinch import PinchProblem

problem = PinchProblem("examples/stream_data/p_illustrative.json", run=True)
problem.export("results/")
```

The repository ships with an Excel template under `Excel_Version/` that matches
the legacy toolchain.

## Documentation

Full documentation (getting started, guides, and API reference) lives under the
`docs/` tree and is designed for Read the Docs. Build it locally with:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

When the site is published the canonical URL will be linked here.

## Testing

Install the project in editable mode along with any optional test dependencies,
then run the test suite with:

```bash
python -m pip install -e .
pytest
```

## Contributing

Issues and pull requests are welcome! Please open a discussion if you have
questions about data formats or feature ideas. When submitting code, aim for:

- Typed interfaces and clear docstrings
- Unit tests covering new behaviour
- Updated documentation where relevant

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
