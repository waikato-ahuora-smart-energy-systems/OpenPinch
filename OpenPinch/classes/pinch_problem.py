# from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

from ..utils import (
    get_problem_from_excel, 
    get_problem_from_csv,
    export_target_summary_to_excel_with_units,
)

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


@dataclass
class PinchProblem:
    """
    A thin, typed orchestrator for loading pinch-analysis input, running targeting,
    and exporting results. Supports:
      • JSON problem files
      • Excel problem files (same format as your existing tool)
      • CSV bundles: a directory containing `streams.csv` and `utilities.csv`
        (or a tuple of two CSV files)
    """
    problem_filepath: Optional[Path] = None
    results_dir: Optional[Path] = None

    # Internal state
    problem_data: Optional[JsonDict] = None
    _results: Optional[JsonDict] = None

    def __init__(
        self,
        problem_filepath: Optional[PathLike] = None,
        results_dir: Optional[PathLike] = None,
        run: bool = True,
    ) -> None:
        if problem_filepath is not None:
            self.problem_filepath = Path(problem_filepath)
            self.load(self.problem_filepath)
        else:
            self.problem_filepath = None
            self._problem_data = None

        self.results_dir = Path(results_dir) if results_dir is not None else None
        self._results = None

        if run:
            try:
                self.target()
            except:
                raise ValueError(
                    "Targeting analysis failed. Check input data format." \
                    "Report any persisent bugs with problem files " \
                    "via a Github issue or email (tim.walmlsey@waikato.ac.nz)"
                )           
                
            if self.results_dir is not None:
                self.export(self.results_dir)


    # ----------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------

    def load(self, source: Union[PathLike, Tuple[PathLike, PathLike]]) -> JsonDict:
        """
        Load input data from one of:
          - JSON file path: *.json
          - Excel file path: *.xlsx / *.xls / *.xlsb / *.xlsm
          - CSV bundle:
              • a directory containing 'streams.csv' and 'utilities.csv', or
              • a tuple: (streams_csv_path, utilities_csv_path)

        Returns the loaded input JSON-like dict.
        """
        if isinstance(source, tuple) and len(source) == 2:
            # CSV tuple form
            streams_csv, utilities_csv = map(Path, source)
            self._problem_data = get_problem_from_csv(streams_csv, utilities_csv, output_json=None)
            self.problem_filepath = None  # Not a single-file source
            return self._problem_data

        src_path = Path(source)

        # 1) JSON
        if src_path.suffix.lower() == ".json":
            try:
                with src_path.open("r", encoding="utf-8") as f:
                    self._problem_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from {src_path}: {e}") from e
            self.problem_filepath = src_path
            return self._problem_data

        # 2) Excel
        elif src_path.suffix.lower() in {".xlsx", ".xls", ".xlsb", ".xlsm"}:
            # Reuse your existing Excel reader; writes options, streams, utilities
            self._problem_data = get_problem_from_excel(src_path, output_json=None)
            self.problem_filepath = src_path
            return self._problem_data

        # 3) CSV bundle via directory lookup
        elif src_path.is_dir():
            streams_csv = src_path / "streams.csv"
            utilities_csv = src_path / "utilities.csv"
            if not streams_csv.exists() or not utilities_csv.exists():
                raise FileNotFoundError(
                    f"CSV directory '{src_path}' must contain 'streams.csv' and 'utilities.csv'."
                )
            self._problem_data = get_problem_from_csv(streams_csv, utilities_csv, output_json=None)
            self.problem_filepath = src_path
            return self._problem_data
        
        raise ValueError(
            f"Unrecognized source '{src_path}'. Provide a JSON/Excel file, "
            f"a directory with 'streams.csv' and 'utilities.csv', or a (streams, utilities) tuple."
        )


    def target(self) -> JsonDict:
        """
        Run the targeting analysis against the loaded input.
        Lazily computes results and caches them.
        """
        if self._problem_data is None:
            raise RuntimeError(
                "No input loaded. Call load(...) first."
            )
        if self._results is None:
            from ..main import pinch_analysis_service
            self._results = pinch_analysis_service(
                self._problem_data
            )
        return self._results


    def export(self, results_dir: Optional[PathLike] = None) -> Path:
        """
        Export the (computed) results to JSON. If not yet computed, will run targeting.
        If results_dir is omitted, uses self.results_dir; otherwise updates it.
        Returns the path written.
        """
        if results_dir is not None:
            self.results_dir = Path(results_dir)

        if self.results_dir is None:
            print(self._results)
            raise ValueError("No results_dir set. Provide a path to export results.")

        # Ensure results exist
        if self._results is None:
            self.target()

        return export_target_summary_to_excel_with_units(self._results, self.results_dir)


    @property
    def problem_data(self) -> Optional[JsonDict]:
        return self._problem_data


    @property
    def results(self) -> Optional[JsonDict]:
        return self._results

    # ----------------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------------

    @classmethod
    def from_json(cls, data: JsonDict) -> "PinchProblem":
        """
        Build directly from an in-memory JSON-like dict.
        """
        obj = cls(problem_filepath=None, results_dir=None, run=False)
        obj._problem_data = data
        return obj


    def to_problem_json(self) -> JsonDict:
        """
        Return the canonical problem JSON (streams/utilities/options).
        """
        if self._problem_data is None:
            raise RuntimeError("No problem_data available. Did you call load(...) or from_json(...)?")
        return self._problem_data
    

    def __repr__(self) -> str:
        src = (
            str(self.problem_filepath)
            if self.problem_filepath is not None
            else "<in-memory or CSV tuple>"
        )
        tgt = str(self.results_dir) if self.results_dir is not None else "<unset>"
        has_results = "yes" if self._results is not None else "no"
        return f"PinchProblem(source={src}, export={tgt}, results={has_results})"