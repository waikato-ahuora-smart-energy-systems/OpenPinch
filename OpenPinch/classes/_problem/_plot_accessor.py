"""Graph-accessor helpers for selecting, rendering, and exporting solved plots."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import pandas as pd

from ...lib.enums import GT
from ...services.common.graph_data import get_output_graph_data
from ...streamlit_webviewer.web_graphing import (
    _build_plotly_graph,
)

if TYPE_CHECKING:
    from ..pinch_problem import PinchProblem

PathLike = Union[str, Path]
GraphRecord = Dict[str, Any]
GraphPayload = Dict[str, GraphRecord]

_GRAPH_TYPE_ALIASES = {
    "etd": GT.ETD.value,
    "energy transfer": GT.ETD.value,
    "energy transfer diagram": GT.ETD.value,
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
    "gcc_x": GT.GCC_X.value,
    "exergy gcc": GT.GCC_X.value,
    "exergetic grand composite curve": GT.GCC_X.value,
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
    "nlp_hp": GT.NLP_HP.value,
    "net load with heat pump": GT.NLP_HP.value,
    "net load curve with heat pump": GT.NLP_HP.value,
    "net load curves with heat pump": GT.NLP_HP.value,
    "net load profile with heat pump": GT.NLP_HP.value,
    "net load profiles with heat pump": GT.NLP_HP.value,
    "net_load_with_heat_pump": GT.NLP_HP.value,
    "nlp_x": GT.NLP_X.value,
    "exergy nlp": GT.NLP_X.value,
    "exergetic net load profiles": GT.NLP_X.value,
    "tsp": GT.TSP.value,
    "total site": GT.TSP.value,
    "total site profiles": GT.TSP.value,
    "sugcc": GT.SUGCC.value,
    "site utility grand composite curve": GT.SUGCC.value,
}


class _PlotAccessor:
    """Callable graph helper that also exposes common named plot shortcuts."""

    def __init__(self, problem: "PinchProblem") -> None:
        """Bind the accessor to one solved or solveable :class:`PinchProblem`."""
        self._problem = problem

    def __call__(self):
        """Return the same graph inventory table as :meth:`catalog`."""
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
        return_graph_data: bool = False,
    ):
        """Select one graph and return either its figure or raw payload."""
        graph_data = self._select_graph(
            zone_name=zone_name,
            graph_type=graph_type,
            index=index,
        )
        if return_graph_data:
            if show:
                raise ValueError(
                    "show=True is only supported when returning a Plotly figure."
                )
            return graph_data
        return self._build_graph_figure(graph_data, show=show)

    def _build_graph_figure(
        self,
        graph_data: GraphRecord,
        *,
        show: bool = False,
    ):
        """Build a Plotly figure from serialized graph data."""
        figure = _build_plotly_graph(graph_data)
        if hasattr(figure, "show") and show:
            figure.show()
        return figure

    def _select_graph(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
        index: int = 0,
    ) -> GraphRecord:
        """Return one graph payload from the filtered graph selection."""
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
    ) -> list[tuple[str, GraphRecord]]:
        """Return all graph payloads matching the optional zone and type filters."""
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
        if getattr(self._problem._results, "graphs", None) is None:
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
            figure = self._build_graph_figure(graph)
            stem = _slugify(f"{graph_zone_name}_{graph.get('type', 'graph')}_{idx}")
            destination = output_path / f"{stem}.html"
            figure.write_html(destination)
            written_paths.append(destination)
        return written_paths

    def composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Composite Curve figure or raw payload."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.CC.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def shifted_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Shifted Composite Curve figure or raw payload."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.SCC.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def balanced_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Balanced Composite Curve figure or raw payload."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.BCC.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Grand Composite Curve figure or raw payload."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def real_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching real temperature GCC payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_R.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def exergetic_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching exergetic GCC output or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_X.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def grand_composite_curve_with_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching GCC with Heat Pump payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.GCC_HP.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def net_load_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching net load profile payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.NLP.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def net_load_profiles_with_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching net load profile (with Heat Pump) or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.NLP_HP.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def exergetic_net_load_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching exergetic NLP payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.NLP_X.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def total_site_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Total Site profiles payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.TSP.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def site_utility_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching site-utility GCC payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.SUGCC.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def energy_transfer_diagram(
        self,
        *,
        zone_name: Optional[str] = None,
        index: float = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching energy-transfer diagram payload or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GT.ETD.value,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )


class _PlotAccessorDescriptor:
    """Non-data descriptor exposing a callable plot accessor on instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        """Return the bound accessor on instances and the descriptor on classes."""
        if obj is None:
            return self
        return _PlotAccessor(obj)


def _normalise_graph_type_selector(graph_type: Optional[str]) -> Optional[str]:
    """Normalize short graph-type aliases to their canonical enum label."""
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
    """Return ``True`` when one graph set matches a user-facing zone selector."""
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
    """Convert a graph-derived label into a filesystem-friendly HTML stem."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "graph"
