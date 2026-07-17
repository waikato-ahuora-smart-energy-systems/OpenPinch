"""Graph-accessor helpers for selecting, rendering, and exporting solved plots."""

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import pandas as pd

from ...analysis.graphs.service import get_output_graph_data
from ...domain.enums import GraphType
from .plotly import build_plotly_figure

if TYPE_CHECKING:
    from ...application.problem import PinchProblem

PathLike = Union[str, Path]
GraphRecord = Dict[str, Any]
GraphData = Dict[str, GraphRecord]


class _PlotAccessor:
    """Read-only graph inventory, selection, rendering, and export helpers."""

    def __init__(self, problem: "PinchProblem") -> None:
        """Bind the accessor to one solved or solveable :class:`PinchProblem`."""
        self._problem = problem

    def catalog(self) -> pd.DataFrame:
        """Return a table describing the available graph outputs."""
        rows = []
        for graph_key, graph_set in self.data().items():
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
        graph_type: GraphType | None = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Select one graph and return either its figure or raw data."""
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
        figure = build_plotly_figure(graph_data)
        if hasattr(figure, "show") and show:
            figure.show()
        return figure

    def _select_graph(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: GraphType | None = None,
        index: int = 0,
    ) -> GraphRecord:
        """Return one graph from the filtered graph selection."""
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
        graph_type: GraphType | None = None,
    ) -> list[tuple[str, GraphRecord]]:
        """Return all graphs matching the optional zone and type filters."""
        graph_data = self.data()
        if graph_type is not None and not isinstance(graph_type, GraphType):
            raise TypeError("graph_type must be a GraphType selected by a plot method.")
        selected_graph_type = graph_type.value if graph_type is not None else None

        zone_items = graph_data.items()
        if zone_name is not None:
            matched_zone_items = [
                (graph_key, graph_set)
                for graph_key, graph_set in graph_data.items()
                if _graph_set_matches_zone_selector(
                    selector=str(zone_name),
                    graph_key=graph_key,
                    graph_set=graph_set,
                )
            ]
            if not matched_zone_items:
                raise KeyError(
                    f"Unknown zone {zone_name!r}. "
                    f"Available zones: {', '.join(graph_data)}"
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

    def data(self) -> GraphData:
        """Return the serialized graph data for the solved problem."""
        if getattr(self._problem._results, "graphs", None) is None:
            raise RuntimeError(
                "No graph data is available. Run a problem.target.<method>() "
                "workflow before requesting plots."
            )

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
        plot: Callable | None = None,
    ) -> list[Path]:
        """Write selected graph outputs as standalone HTML files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        selected = self._select_graphs(
            zone_name=zone_name,
            graph_type=self._graph_type_for_method(plot),
        )
        written_paths = []
        for idx, (graph_zone_name, graph) in enumerate(selected, start=1):
            figure = self._build_graph_figure(graph)
            stem = _slugify(f"{graph_zone_name}_{graph.get('type', 'graph')}_{idx}")
            destination = output_path / f"{stem}.html"
            figure.write_html(destination)
            written_paths.append(destination)
        return written_paths

    def export_gallery(
        self,
        output_dir: PathLike,
        *,
        zone_name: Optional[str] = None,
        plot: Callable | None = None,
        index_name: str = "index.html",
    ) -> Path:
        """Write selected graphs and a browsable HTML index to ``output_dir``."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        selected = self._select_graphs(
            zone_name=zone_name,
            graph_type=self._graph_type_for_method(plot),
        )
        links = []
        for idx, (graph_zone_name, graph) in enumerate(selected, start=1):
            figure = self._build_graph_figure(graph)
            graph_title = str(graph.get("name") or graph.get("type") or f"Graph {idx}")
            stem = _slugify(f"{graph_zone_name}_{graph.get('type', 'graph')}_{idx}")
            destination = output_path / f"{stem}.html"
            figure.write_html(destination)
            links.append((destination.name, graph_zone_name, graph_title))

        index_path = output_path / index_name
        index_path.write_text(_gallery_index_html(links), encoding="utf-8")
        return index_path

    def _graph_type_for_method(self, plot: Callable | None) -> GraphType | None:
        if plot is None:
            return None
        if not callable(plot):
            raise TypeError("plot must be one of the bound problem.plot methods.")
        owner = getattr(plot, "__self__", None)
        name = getattr(plot, "__name__", "")
        if not isinstance(owner, _PlotAccessor) or owner._problem is not self._problem:
            raise ValueError(
                "plot must be a method bound to this problem.plot accessor."
            )
        method_types = {
            "composite_curve": GraphType.CC,
            "shifted_composite_curve": GraphType.SCC,
            "balanced_composite_curve": GraphType.BCC,
            "grand_composite_curve": GraphType.GCC,
            "real_grand_composite_curve": GraphType.GCC_R,
            "exergetic_grand_composite_curve": GraphType.GCC_X,
            "grand_composite_curve_with_heat_pump": GraphType.GCC_HP,
            "net_load_profiles": GraphType.NLP,
            "net_load_profiles_with_heat_pump": GraphType.NLP_HP,
            "exergetic_net_load_profiles": GraphType.NLP_X,
            "total_site_profiles": GraphType.TSP,
            "site_utility_grand_composite_curve": GraphType.SUGCC,
            "energy_transfer_diagram": GraphType.ETD,
        }
        try:
            return method_types[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported plot method {name!r}.") from exc

    def composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Composite Curve figure or raw data."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.CC,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def shifted_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Shifted Composite Curve figure or raw data."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.SCC,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def balanced_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Balanced Composite Curve figure or raw data."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.BCC,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Grand Composite Curve figure or raw data."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.GCC,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def real_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching real temperature GCC data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.GCC_R,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def exergetic_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching exergetic GCC output or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.GCC_X,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def grand_composite_curve_with_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching GCC with Heat Pump data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.GCC_HP,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def net_load_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching net load profile data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.NLP,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def net_load_profiles_with_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching net load profile (with Heat Pump) or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.NLP_HP,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def exergetic_net_load_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching exergetic NLP data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.NLP_X,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def total_site_profiles(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching Total Site profiles data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.TSP,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def site_utility_grand_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching site-utility GCC data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.SUGCC,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )

    def energy_transfer_diagram(
        self,
        *,
        zone_name: Optional[str] = None,
        index: int = 0,
        show: bool = False,
        return_graph_data: bool = False,
    ):
        """Return the first matching energy-transfer diagram data or figure."""
        return self._plot_graph(
            zone_name=zone_name,
            graph_type=GraphType.ETD,
            index=index,
            show=show,
            return_graph_data=return_graph_data,
        )


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


def _gallery_index_html(links: list[tuple[str, str, str]]) -> str:
    """Return a small static graph-gallery index page."""
    items = "\n".join(
        "<li>"
        f'<a href="{escape(filename)}">{escape(title)}</a>'
        f" <span>{escape(zone)}</span>"
        "</li>"
        for filename, zone, title in links
    )
    if not items:
        items = "<li>No graphs matched the requested selection.</li>"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>OpenPinch Graph Gallery</title>\n"
        "  <style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5}"
        "span{color:#666;margin-left:.5rem}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>OpenPinch Graph Gallery</h1>\n"
        f"  <ul>{items}</ul>\n"
        "</body>\n"
        "</html>\n"
    )
