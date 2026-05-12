import pandas as pd

from typing import Optional, Any, Union, Dict, TYPE_CHECKING
from pathlib import Path

from ...lib.enums import GT
from ...streamlit_webviewer.web_graphing import (
    _build_plotly_graph,
)
from ...services.common.graph_data import get_output_graph_data

if TYPE_CHECKING:
    from ...classes.pinch_problem import PinchProblem

PathLike = Union[str, Path]
GraphPayload = Dict[str, Dict[str, Any]]

_GRAPH_TYPE_ALIASES = {
    "cc": GT.CC.value,
    "composite": GT.CC.value,
    "composite curve": GT.CC.value,
    "composite curves": GT.CC.value,
    "scc": GT.SCC.value,
    "shifted": GT.SCC.value,
    "shifted composite": GT.SCC.value,
    "shifted composite curve": GT.SCC.value,
    "shifted composite curves": GT.SCC.value,
    "bcc": GT.BCC.value,
    "balanced": GT.BCC.value,
    "balanced composite": GT.BCC.value,
    "balanced composite curve": GT.BCC.value,
    "balanced composite curves": GT.BCC.value,
    "gcc": GT.GCC.value,
    "grand composite": GT.GCC.value,
    "grand composite curve": GT.GCC.value,
    "nlc": GT.NLP.value,
    "net load": GT.NLP.value,
    "net load curve": GT.NLP.value,
    "net load curves": GT.NLP.value,
    "net load profile": GT.NLP.value,
    "net load profiles": GT.NLP.value,
    "net_load": GT.NLP.value,
    "net_load curve": GT.NLP.value,
    "net_load curves": GT.NLP.value,
    "net_load profile": GT.NLP.value,
    "net_load profiles": GT.NLP.value,
    "tsp": GT.TSP.value,
    "total site": GT.TSP.value,
    "total site profiles": GT.TSP.value,
    "sugcc": GT.SUGCC.value,
    "site utility grand composite curve": GT.SUGCC.value,
}


class _PlotAccessor:
    """Callable graph helper that also exposes common named plot shortcuts."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def __call__(self):
        return self.catalog()

    def catalog(self) -> pd.DataFrame:
        """Return a table describing the available graph outputs."""
        rows = []
        for graph_key, graph_set in self.get_graph_data().items():
            graph_zone_name = graph_set.get("zone_name", graph_key)
            graph_zone_address = graph_set.get("zone_address", graph_zone_name)
            target_name = graph_set.get("name", graph_key)
            target_type = graph_set.get("target_type")
            for index, graph in enumerate(graph_set.get("graphs", [])):
                rows.append(
                    {
                        "Zone": graph_zone_name,
                        "Zone Address": graph_zone_address,
                        "Target": target_name,
                        "Target Type": target_type,
                        "Graph Type": graph.get("type"),
                        "Graph Name": graph.get("name", f"Graph {index + 1}"),
                        "Index": index,
                    }
                )
        return pd.DataFrame(rows)

    def _plot_graph(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
        index: int = 0,
        show: bool = False,
    ):
        """Build a Plotly figure for one graph from the solved result set."""
        graph = self._select_graph(
            zone_name=zone_name,
            graph_type=graph_type,
            index=index,
        )
        figure = _build_plotly_graph(graph)
        if hasattr(figure, "show") and show:
            figure.show()
        return figure

    def _select_graph(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
        index: int = 0,
    ) -> dict:
        graphs = self._select_graphs(zone_name=zone_name, graph_type=graph_type)
        if not graphs:
            raise ValueError("No graphs matched the requested selection.")
        try:
            return graphs[index][1]
        except IndexError as exc:
            raise IndexError(
                f"Graph index {index} is out of range for the selected graphs."
            ) from exc

    def _select_graphs(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        payload = self.get_graph_data()
        selected_graph_type = _normalise_graph_type_selector(graph_type)

        zone_items = payload.items()
        if zone_name is not None:
            matched_zone_items = [
                (graph_key, graph_set)
                for graph_key, graph_set in payload.items()
                if _graph_set_matches_zone_selector(
                    selector=str(zone_name),
                    graph_key=graph_key,
                    graph_set=graph_set,
                )
            ]
            if not matched_zone_items:
                raise KeyError(
                    f"Unknown zone {zone_name!r}. Available zones: {', '.join(payload)}"
                )
            zone_items = matched_zone_items

        selected = []
        for graph_key, graph_set in zone_items:
            zone_identifier = graph_set.get("zone_address") or graph_set.get(
                "zone_name", graph_key
            )
            for graph in graph_set.get("graphs", []):
                if (
                    selected_graph_type is not None
                    and graph.get("type") != selected_graph_type
                ):
                    continue
                selected.append((zone_identifier, graph))
        return selected

    def get_graph_data(self) -> GraphPayload:
        """Return the serialized graph payload for the solved problem."""
        if getattr(self._problem._results, "graphs", None) == None:
            self._problem.target()

        graphs = getattr(self._problem._results, "graphs", None)

        if graphs:
            return {
                key: value.model_dump() if hasattr(value, "model_dump") else dict(value)
                for key, value in graphs.items()
            }
        return get_output_graph_data(self._problem._master_zone)

    def export(
        self,
        output_dir: PathLike,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
    ) -> list[Path]:
        """Write selected graph outputs as standalone HTML files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        selected = self._select_graphs(zone_name=zone_name, graph_type=graph_type)
        written_paths = []
        for idx, (graph_zone_name, graph) in enumerate(selected, start=1):
            figure = _build_plotly_graph(graph)
            stem = _slugify(f"{graph_zone_name}_{graph.get('type', 'graph')}_{idx}")
            destination = output_path / f"{stem}.html"
            figure.write_html(destination)
            written_paths.append(destination)
        return written_paths

    def composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.CC.value,
            index=index,
            show=show,
        )

    def shifted_composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching shifted-composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.SCC.value,
            index=index,
            show=show,
        )

    def balanced_composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching balanced-composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.BCC.value,
            index=index,
            show=show,
        )

    def grand_composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching grand-composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC.value,
            index=index,
            show=show,
        )

    def real_grand_composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching grand-composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_R.value,
            index=index,
            show=show,
        )

    def grand_composite_curve_with_heat_pump(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching GCC-with-heat-pump overlay figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_HP.value,
            index=index,
            show=show,
        )

    def net_load_profiles(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching net-load-profile figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.NLP.value,
            index=index,
            show=show,
        )

    def real_grand_composite_curve(
        self, *, zone_name: Optional[str] = None, index: float = 0, show: bool = False
    ):
        """Build the first matching grand-composite-curve figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_R.value,
            index=index,
            show=show,
        )


class _PlotAccessorDescriptor:
    """Non-data descriptor exposing a callable plot accessor on instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _PlotAccessor(obj)


def _normalise_graph_type_selector(graph_type: Optional[str]) -> Optional[str]:
    if graph_type is None:
        return None
    text = str(graph_type).strip()
    return _GRAPH_TYPE_ALIASES.get(text.lower(), text)


def _graph_set_matches_zone_selector(
    *,
    selector: str,
    graph_key: str,
    graph_set: dict[str, Any],
) -> bool:
    text = str(selector).strip()
    suffix = text.split("/", 1)[-1]
    candidates = (
        graph_key,
        graph_set.get("name"),
        graph_set.get("zone_name"),
        graph_set.get("zone_address"),
        graph_set.get("target_type"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        candidate_text = str(candidate)
        if text == candidate_text or suffix == candidate_text:
            return True
        if "/" in candidate_text and suffix == candidate_text.split("/", 1)[-1]:
            return True
    return False


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "graph"
