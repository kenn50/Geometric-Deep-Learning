#Created by GPT 6.1 Sol

"""Browser-based comparison and expansion of radius and kNN graphs."""

import asyncio
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import panel as pn
import param
import plotly.graph_objects as go
from plotly.basedatatypes import Undefined
from plotly.subplots import make_subplots
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.neighbors import NearestNeighbors


def hop_distances(graph, seed, levels):
    """Shortest hop distances up to levels; -1 marks unreached points."""
    graph = graph.tocsr()
    if not 0 <= seed < graph.shape[0] or levels < 0:
        raise ValueError("Invalid seed or expansion depth.")
    distances = np.full(graph.shape[0], -1, dtype=np.int32)
    distances[seed] = 0
    queue = deque([seed])
    while queue:
        node = queue.popleft()
        if distances[node] >= levels:
            continue
        for neighbor in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _edge_coordinates(points, rows, columns):
    segments = np.full((len(rows), 3, 3), np.nan, dtype=np.float32)
    segments[:, 0] = points[rows]
    segments[:, 1] = points[columns]
    return segments.reshape(-1, 3)


def _neighbor_graph(index: NearestNeighbors, kind, value):
    """Build a SciPy CSR graph using a fitted sklearn NearestNeighbors object.

    ``index`` is the reusable neighbor-search object, not a point's integer ID.
    Its radius_neighbors_graph / kneighbors_graph methods construct the edges.
    """
    if kind == "radius":
        graph = index.radius_neighbors_graph(radius=value, mode="connectivity")
    else:
        graph = index.kneighbors_graph(n_neighbors=value, mode="connectivity")
    graph = graph.maximum(graph.T).astype(np.uint8).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


@dataclass
class ConnectivitySearch:
    value: float | int
    lower_bound: float | int
    graph: object
    evaluations: int


def find_connected_radius(index, initial_radius=0.05, tolerance=1e-4):
    """Bracket and bisect the connectivity threshold; return the connected bound.

    The lower bound is disconnected, except when radius zero is already
    connected. The interval width is at most tolerance, subject to floating
    point precision. Search candidates never create visualization geometry.
    """
    if not np.isfinite(initial_radius) or initial_radius < 0:
        raise ValueError("Initial radius must be finite and nonnegative.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Radius tolerance must be positive and finite.")
    evaluations = 0

    def evaluate(radius):
        nonlocal evaluations
        graph = _neighbor_graph(index, "radius", radius)
        evaluations += 1
        return connected_components(graph, directed=False, return_labels=False) == 1, graph

    connected, graph = evaluate(0.0)
    if connected:
        return ConnectivitySearch(0.0, 0.0, graph, evaluations)
    lower, upper = 0.0, max(float(initial_radius), tolerance)
    for _ in range(128):
        connected, graph = evaluate(upper)
        if connected:
            upper_graph = graph
            break
        lower, upper = upper, upper * 2
        if not np.isfinite(upper):
            raise ValueError("Could not find a finite connected radius.")
    else:
        raise ValueError("Could not bracket a connected radius.")

    while upper - lower > tolerance:
        middle = lower + (upper - lower) / 2
        if middle == lower or middle == upper:
            break
        connected, graph = evaluate(middle)
        if connected:
            upper, upper_graph = middle, graph
        else:
            lower = middle
    return ConnectivitySearch(upper, lower, upper_graph, evaluations)


def find_connected_k(index, initial_k=10):
    """Find the exact minimum k for the symmetric union-kNN graph."""
    maximum = index.n_samples_fit_ - 1
    if maximum < 1 or not isinstance(initial_k, (int, np.integer)) or initial_k < 1:
        raise ValueError("k search requires at least two points and a positive integer k.")
    evaluations = 0

    def evaluate(k):
        nonlocal evaluations
        graph = _neighbor_graph(index, "knn", k)
        evaluations += 1
        return connected_components(graph, directed=False, return_labels=False) == 1, graph

    connected, graph = evaluate(1)
    if connected:
        return ConnectivitySearch(1, 0, graph, evaluations)
    lower, upper = 1, min(max(2, int(initial_k)), maximum)
    while True:
        connected, graph = evaluate(upper)
        if connected:
            upper_graph = graph
            break
        if upper == maximum:
            raise ValueError("Could not find a connected kNN graph.")
        lower, upper = upper, min(upper * 2, maximum)

    while upper - lower > 1:
        middle = (lower + upper) // 2
        connected, graph = evaluate(middle)
        if connected:
            upper, upper_graph = middle, graph
        else:
            lower = middle
    return ConnectivitySearch(upper, lower, upper_graph, evaluations)


class GraphExplorer:
    """One browser session, with shared seed and independent graph expansions."""

    def __init__(self, points, radius=0.05, k=10):
        self.points = points
        self.seed = None
        self.graphs = {}
        self.edges = {}
        self.distances = {}
        self._edge_xyz = {}
        self._expansion_cache = {}
        self._graph_values = {}
        self._last_expansion = {}
        self._trace_nodes = {}
        self.search_results = {}
        self._search_running = False
        # Fit sklearn's neighbor search once; reuse it for every radius/k query.
        self._neighbor_search = NearestNeighbors(algorithm="kd_tree", n_jobs=1).fit(points)

        self.radius = pn.widgets.FloatSlider(
            name="Radius", start=min(0.005, radius), end=max(0.2, radius),
            step=0.005, value=radius, width=280,
        )
        self.k = pn.widgets.IntSlider(
            name="k nearest neighbors", start=1,
            end=min(max(50, k), len(points) - 1), value=k, width=280,
        )
        self.search_radius = pn.widgets.Button(name="Binary search radius", button_type="primary")
        self.search_k = pn.widgets.Button(name="Binary search k", button_type="primary")
        self.radius_buffer = pn.widgets.FloatInput(
            name="Radius buffer beta (0.1 = +10%)", value=0.1, start=0, step=0.05, width=240,
        )
        self.radius_tolerance = pn.widgets.FloatInput(
            name="Radius search tolerance", value=1e-4, start=1e-10, step=1e-4, width=200,
        )
        self.radius_search_status = pn.pane.Markdown("")
        self.k_search_status = pn.pane.Markdown("")
        self.step = pn.widgets.IntInput(
            name="Levels per Expand click", value=1, start=1, end=20, width=180,
        )
        self.depth = pn.widgets.IntSlider(
            name="Visible expansion depth", start=0, end=20, value=0, width=300,
        )
        self.expand = pn.widgets.Button(name="Expand", button_type="primary", disabled=True)
        self.clear = pn.widgets.Button(name="Clear selection")
        self.selection = pn.pane.Markdown("**Click a point in either graph to select a seed.**")
        self.summary = pn.pane.Markdown("")
        self.error = pn.pane.Alert("", alert_type="danger", visible=False)

        self.figure = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=["Radius graph", "kNN graph"], horizontal_spacing=0.03,
        )
        for column in (1, 2):
            # Stable trace order: edges, points, red edges, red points, seed.
            traces = [
                go.Scatter3d(mode="lines", line=dict(color="black", width=2),
                             opacity=0.45, hoverinfo="skip"),
                go.Scatter3d(mode="markers", marker=dict(size=3, coloraxis="coloraxis"),
                             hovertemplate="Point %{customdata[0]}<br>Neighbors: %{customdata[1]}<extra></extra>"),
                go.Scatter3d(mode="lines", line=dict(color="red", width=5), hoverinfo="skip"),
                go.Scatter3d(mode="markers", marker=dict(size=5, colorscale="Reds", cmin=0,
                             showscale=False),
                             hovertemplate="Point %{customdata[0]}<br>Hop distance: %{customdata[1]}<extra></extra>"),
                go.Scatter3d(mode="markers", marker=dict(size=10, color="gold", symbol="diamond",
                             line=dict(color="black", width=2)),
                             hovertemplate="Seed: %{customdata[0]}<extra></extra>"),
            ]
            for trace in traces:
                trace.update(x=[], y=[], z=[], showlegend=False)
                self.figure.add_trace(trace, row=1, col=column)

        lower, upper = points.min(axis=0), points.max(axis=0)
        padding = 0.05 * np.maximum(upper - lower, 1e-6)
        self.figure.update_scenes(
            aspectmode="data", dragmode="orbit", uirevision="graph-camera",
            xaxis=dict(range=[lower[0] - padding[0], upper[0] + padding[0]]),
            yaxis=dict(range=[lower[1] - padding[1], upper[1] + padding[1]]),
            zaxis=dict(range=[lower[2] - padding[2], upper[2] + padding[2]]),
        )
        self.figure.update_layout(
            height=700, uirevision="graph-camera", margin=dict(l=0, r=0, b=0, t=90),
            paper_bgcolor="white", coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1,
                                                 colorbar=dict(title="Neighbors")),
        )
        self.plot = pn.pane.Plotly(
            self.figure, height=700, sizing_mode="stretch_width", link_figure=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )
        # Plotly batches contain Undefined sentinels for unchanged properties.
        # Panel forwards these through Bokeh, whose serializer cannot encode
        # them. Fill those slots with the existing value before transport.
        self._send_plot_update = self.figure._send_update_msg
        self.figure._send_update_msg = self._send_batched_update
        self.plot.param.watch(self._on_click, "click_data")
        self.radius.param.watch(lambda event: self._rebuild("radius"), "value_throttled")
        self.k.param.watch(lambda event: self._rebuild("knn"), "value_throttled")
        # Dragging the depth slider does not send a redraw for every tick.
        self.depth.param.watch(lambda event: self._refresh_expansion(), "value_throttled")
        self.expand.on_click(self._on_expand)
        self.clear.on_click(self._on_clear)
        self.search_radius.on_click(self._on_search_radius)
        self.search_k.on_click(self._on_search_k)

        self.panel = pn.Column(
            "# Training-data graph expansion",
            pn.Row(self.radius, self.search_radius, self.radius_buffer, self.radius_tolerance),
            self.radius_search_status,
            pn.Row(self.k, self.search_k),
            self.k_search_status,
            pn.Row(self.step, self.expand, self.clear, self.depth),
            self.selection, self.summary, self.error,
            "Gold diamond: seed. Red points: reached nodes. Red lines: graph edges within the reached region. "
            "The same seed is expanded independently in both graphs.",
            self.plot, sizing_mode="stretch_width",
        )
        with self.figure.batch_update():
            self._rebuild("radius")
            self._rebuild("knn")

    def _patch(self, index, xyz, **style):
        # Property assignments participate in batch_update. Explicit calls to
        # plotly_restyle send separate messages even inside a batch context.
        trace = self.figure.data[index]
        trace.x, trace.y, trace.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        for key, value in style.items():
            trace[key] = value

    def _send_batched_update(self, restyle_data, relayout_data, trace_indexes=None,
                             source_view_id=None):
        indices = list(range(len(self.figure.data))) if trace_indexes is None else trace_indexes
        normalized = {}
        for key, values in restyle_data.items():
            if isinstance(values, list):
                normalized[key] = [
                    self.figure.data[indices[i]][key] if value is Undefined else value
                    for i, value in enumerate(values)
                ]
            else:
                normalized[key] = values
        self._send_plot_update(
            restyle_data=normalized, relayout_data=relayout_data,
            trace_indexes=trace_indexes, source_view_id=source_view_id,
        )

    def _rebuild(self, kind, graph=None):
        value = self.radius.value if kind == "radius" else self.k.value
        if self._graph_values.get(kind) == value:
            return
        self.plot.loading = True
        try:
            self.error.visible = False
            if graph is None:
                graph = _neighbor_graph(self._neighbor_search, kind, value)
            self.graphs[kind] = graph
            self._graph_values[kind] = value
            self._expansion_cache.pop(kind, None)
            self._last_expansion.pop(kind, None)
            upper = triu(graph, k=1).tocoo()
            self.edges[kind] = (upper.row, upper.col)
            # Build all line segments once, only when the graph changes.
            self._edge_xyz[kind] = _edge_coordinates(
                self.points, upper.row, upper.col
            ).reshape(-1, 3, 3)
            base = 0 if kind == "radius" else 5
            degrees = graph.getnnz(axis=1)
            ids = np.arange(len(self.points))
            self._trace_nodes[base + 1] = ids
            name = f"Radius graph: r = {self.radius.value:.6g}" if kind == "radius" else f"kNN graph: k = {self.k.value}"
            components = connected_components(graph, directed=False, return_labels=False)
            with self.figure.batch_update():
                self._patch(base, self._edge_xyz[kind].reshape(-1, 3))
                self._patch(base + 1, self.points, **{
                    "marker.color": degrees, "customdata": np.column_stack((ids, degrees)),
                })
                self.figure.layout.annotations[base // 5].text = (
                    f"{name}<br>Connected: {'yes' if components == 1 else 'no'}"
                    f" | Components: {components}<br>Mean neighbors: {degrees.mean():.1f}"
                )
                self.figure.layout.coloraxis.cmax = max(
                    1, *(int(g.getnnz(axis=1).max()) for g in self.graphs.values())
                )
                self._refresh_expansion()
        except Exception as exc:
            self.error.object = f"Graph update failed: {type(exc).__name__}: {exc}"
            self.error.visible = True
            raise
        finally:
            self.plot.loading = False

    async def _on_search_radius(self, event):
        await self._run_search("radius")

    async def _on_search_k(self, event):
        await self._run_search("knn")

    def _search(self, kind, initial_value, tolerance, beta):
        if kind == "radius":
            if not np.isfinite(beta) or beta < 0:
                raise ValueError("Radius buffer beta must be finite and nonnegative.")
            result = find_connected_radius(self._neighbor_search, initial_value, tolerance)
            applied = result.value * (1 + beta)
            if not np.isfinite(applied):
                raise ValueError("The buffered radius is not finite.")
            graph = result.graph if applied == result.value else _neighbor_graph(self._neighbor_search, kind, applied)
        else:
            result = find_connected_k(self._neighbor_search, initial_value)
            applied, graph = result.value, result.graph
        return result, applied, graph

    async def _run_search(self, kind):
        if self._search_running:
            return
        self._search_running = True
        controls = (self.radius, self.k, self.search_radius, self.search_k,
                    self.radius_buffer, self.radius_tolerance)
        previous_disabled = [widget.disabled for widget in controls]
        for widget in controls:
            widget.disabled = True
        button = self.search_radius if kind == "radius" else self.search_k
        status = self.radius_search_status if kind == "radius" else self.k_search_status
        button.loading = True
        status.object = "Searching for one connected component..."
        self.error.visible = False
        initial = self.radius.value if kind == "radius" else self.k.value
        tolerance, beta = self.radius_tolerance.value, self.radius_buffer.value
        started = perf_counter()
        try:
            # Test adjacency only in a worker; draw the final graph just once.
            result, applied, graph = await asyncio.to_thread(
                self._search, kind, initial, tolerance, beta
            )
            slider = self.radius if kind == "radius" else self.k
            slider.start = min(slider.start, applied)
            slider.end = max(slider.end, applied)
            if kind == "radius":
                slider.step = min(slider.step, tolerance)
            with param.discard_events(slider), param.edit_constant(slider):
                slider.value = applied
                slider.value_throttled = applied
            self._rebuild(kind, graph=graph)
            self.search_results[kind] = result
            timing = f"{result.evaluations} connectivity checks; {perf_counter() - started:.2f} s."
            if kind == "radius":
                status.object = (
                    f"**Connectivity threshold:** {result.value:.8g} "
                    f"(bracket [{result.lower_bound:.8g}, {result.value:.8g}]). "
                    f"**Applied:** {applied:.8g} = threshold × (1 + {beta:g}). {timing}"
                )
            else:
                status.object = f"**Minimum connected k:** {result.value} (symmetric union-kNN). {timing}"
        except Exception as exc:
            self.error.object = f"Connectivity search failed: {type(exc).__name__}: {exc}"
            self.error.visible = True
            status.object = "Search failed; see the error below."
        finally:
            button.loading = False
            for widget, disabled in zip(controls, previous_disabled):
                widget.disabled = disabled
            self._search_running = False

    def select_seed(self, seed):
        """Select an original X_train row; selection is shared across both views."""
        seed = int(seed)
        if not 0 <= seed < len(self.points):
            raise ValueError("Point index is out of range.")
        self.seed = seed
        self.expand.disabled = False
        self._set_depth(0)

    def _set_depth(self, value):
        # Buttons and point clicks change both slider values together. Suppress
        # the watcher so each action results in exactly one refresh.
        with param.discard_events(self.depth), param.edit_constant(self.depth):
            self.depth.value = value
            self.depth.value_throttled = value
        self._refresh_expansion()

    def _on_click(self, event):
        data = event.new or {}
        for point in data.get("points", []):
            trace = point.get("curveNumber")
            number = point.get("pointNumber", point.get("pointIndex"))
            ids = self._trace_nodes.get(trace)
            if ids is not None and isinstance(number, (int, np.integer)) and 0 <= number < len(ids):
                self.select_seed(ids[number])
                break

    def _on_expand(self, event):
        if self.seed is not None:
            target = self.depth.value + self.step.value
            self.depth.end = max(self.depth.end, target)
            self._set_depth(target)

    def _on_clear(self, event):
        self.seed = None
        self.expand.disabled = True
        self._set_depth(0)

    def _cached_expansion(self, kind):
        cache = self._expansion_cache.get(kind)
        if cache is not None and cache["seed"] == self.seed:
            return cache

        # Calculate all hop distances once per selected seed and graph. scipy
        # performs the traversal in compiled code rather than Python loops.
        raw = dijkstra(self.graphs[kind], indices=self.seed, unweighted=True)
        full = np.full(len(self.points), -1, dtype=np.int32)
        finite = np.isfinite(raw)
        full[finite] = raw[finite].astype(np.int32)

        nodes = np.flatnonzero(full > 0)
        nodes = nodes[np.argsort(full[nodes], kind="stable")]
        node_hops = full[nodes]

        rows, columns = self.edges[kind]
        edge_ids = np.flatnonzero((full[rows] >= 0) & (full[columns] >= 0))
        edge_hops = np.maximum(full[rows[edge_ids]], full[columns[edge_ids]])
        order = np.argsort(edge_hops, kind="stable")
        edge_ids, edge_hops = edge_ids[order], edge_hops[order]

        cache = {
            "seed": self.seed, "full": full, "nodes": nodes,
            "node_hops": node_hops, "node_xyz": self.points[nodes],
            "customdata": np.column_stack((nodes, node_hops)),
            "edge_hops": edge_hops,
            # A depth change now takes a prefix view; no new line geometry.
            "edge_xyz": self._edge_xyz[kind][edge_ids],
            "level_counts": np.bincount(full[finite]),
        }
        self._expansion_cache[kind] = cache
        return cache

    def _refresh_expansion(self):
        counts = []
        empty = np.empty((0, 3), dtype=np.float32)
        depth = self.depth.value
        with self.figure.batch_update():
            for kind, graph in self.graphs.items():
                base = 0 if kind == "radius" else 5
                if self.seed is None:
                    self.distances[kind] = np.full(len(self.points), -1, dtype=np.int32)
                    render_key = (None,)
                    if self._last_expansion.get(kind) != render_key:
                        self._patch(base + 2, empty)
                        for index in (base + 3, base + 4):
                            self._trace_nodes[index] = np.empty(0, dtype=int)
                            self._patch(index, empty, customdata=np.empty((0, 2), dtype=int))
                    self._last_expansion[kind] = render_key
                    counts.append(f"**{kind}:** 0 reached")
                    continue

                cache = self._cached_expansion(kind)
                node_count = int(np.searchsorted(cache["node_hops"], depth, side="right"))
                edge_count = int(np.searchsorted(cache["edge_hops"], depth, side="right"))
                full = cache["full"]
                self.distances[kind] = np.where(full <= depth, full, -1)
                render_key = (self.seed, node_count, edge_count)
                if self._last_expansion.get(kind) != render_key:
                    self._patch(base + 2, cache["edge_xyz"][:edge_count].reshape(-1, 3))
                    self._trace_nodes[base + 3] = cache["nodes"][:node_count]
                    self._patch(base + 3, cache["node_xyz"][:node_count], **{
                        "customdata": cache["customdata"][:node_count],
                        "marker.color": cache["node_hops"][:node_count],
                        "marker.cmax": max(1, len(cache["level_counts"]) - 1),
                    })
                    seeds = np.array([self.seed], dtype=int)
                    self._trace_nodes[base + 4] = seeds
                    self._patch(base + 4, self.points[seeds], customdata=seeds.reshape(-1, 1))
                    self._last_expansion[kind] = render_key
                frontier = int(cache["level_counts"][depth]) if depth < len(cache["level_counts"]) else 0
                counts.append(f"**{kind}:** {node_count + 1:,} reached; {frontier:,} at depth {depth}")
        self.selection.object = (
            f"**Selected point: {self.seed}** | Visible depth: **{self.depth.value}**"
            if self.seed is not None else "**Click a point in either graph to select a seed.**"
        )
        self.summary.object = " | ".join(counts)


@dataclass
class GraphViewer:
    server: object
    url: str
    sessions: list

    def stop(self):
        self.server.stop()
        self.server.join(timeout=3)


def open_graph_view(x_train, *, radius=0.05, k=10, port=5006, open_browser=True):
    """Serve an interactive radius/kNN graph viewer on localhost."""
    if hasattr(x_train, "detach"):
        x_train = x_train.detach().cpu().numpy()
    points = np.array(x_train, dtype=np.float32, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("x_train must have shape (N, 3), with N >= 2.")
    if not np.isfinite(points).all():
        raise ValueError("x_train must contain finite coordinates.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k < len(points):
        raise ValueError("k must be an integer between 1 and N - 1.")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535.")
    pn.extension("plotly")
    sessions = []

    def app():
        try:
            explorer = GraphExplorer(points, radius=radius, k=int(k))
            sessions.append(explorer)
            pn.state.on_session_destroyed(lambda context: sessions.remove(explorer) if explorer in sessions else None)
            return explorer.panel
        except Exception as exc:
            return pn.pane.Alert(f"Viewer initialization failed: {type(exc).__name__}: {exc}", alert_type="danger")

    server = pn.serve(
        app, address="127.0.0.1", port=port,
        websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"],
        threaded=True, show=open_browser, title="Graph expansion",
    )
    return GraphViewer(server, f"http://127.0.0.1:{port}", sessions)
