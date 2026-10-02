"""Explore the two-dimensional charts of a trained :class:`A_CAE.CAE`.

Example::

    from CustomModules.viser_viz import open_split_view

    viewer = open_split_view(X_train, cae)
    # The browser opens at viewer.url. Call viewer.stop() when finished.
    # Individual servers can be stopped with viewer.stop_latent_server(),
    # viewer.stop_x_server(), or viewer.stop_split_server().

``X_train`` is an ``(N, 3)`` point cloud. ``cae`` must have ``x_dim == 3`` and
``z_dim == 2`` and provide ``encode(x, chart_index)`` and
``decode(z, chart_index)``. Those methods receive batched torch tensors of
shapes ``(B, 3)`` and ``(B, 2)`` and return tensors of shapes ``(B, 2)`` and
``(B, 3)`` respectively. The epsilon control uses squared Euclidean
reconstruction error, matching ``A_CAE.norm_squared``.

Observation points and latent posterior points are colored by the chart with
the lowest reconstruction error for that input. The draggable marker uses the
best chart for its current decoded point. Hovering a chart in the latent Viser
GUI highlights its winning observation points. The observation marker uses the
same changing chart color.
"""

from __future__ import annotations

import threading
import colorsys
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import viser

from CustomModules.viser_split_view import SceneView, ViserSplitView


_CHART_CONTROLS_HTML = """
<style>
  html, body { margin: 0; height: 100%; background: transparent; }
  #chart-picker { width: 100%; height: 100%; overflow: auto; box-sizing: border-box;
    padding: 8px; border: 1px solid #4a5060; border-radius: 6px;
    background: #191d25; color: #f0f3f8; font: 13px system-ui, sans-serif; }
  #chart-picker strong { display: block; margin: 0 0 7px; }
  #chart-picker small { display: block; color: #acb5c5; margin-bottom: 8px; }
  #chart-list { display: grid; gap: 3px; }
  .chart-row { display: flex; align-items: center; width: 100%; gap: 8px;
    padding: 6px 8px; color: inherit; background: transparent; border: 1px solid transparent;
    border-radius: 5px; text-align: left; font: inherit; cursor: pointer; }
  .chart-row:hover, .chart-row:focus-visible { background: #424b5d; }
  .chart-row.active { border-color: #c4cbd6; }
  .chart-swatch { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }
  .chart-error { margin-left: auto; color: #c5cbd6; font-variant-numeric: tabular-nums; }
</style>
<aside id="chart-picker" aria-label="Charts ranked by reconstruction error">
  <strong>Charts - best reconstruction first</strong>
  <small>Hover to highlight points. Click to switch charts.</small>
  <div id="chart-list"></div>
</aside>
<script>
  const chartList = document.getElementById('chart-list');
  let lastState = '';
  async function chartRequest(path) {
    const response = await fetch(path, {cache: 'no-store'});
    if (!response.ok) throw new Error(`Chart request failed: ${response.status}`);
    return response.json();
  }
  function renderCharts(state) {
    const signature = JSON.stringify(state);
    if (signature === lastState) return;
    lastState = signature;
    chartList.replaceChildren();
    for (const chart of state.charts) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'chart-row' + (chart.index === state.active ? ' active' : '');
      row.dataset.chart = String(chart.index);
      const swatch = document.createElement('span');
      swatch.className = 'chart-swatch';
      swatch.style.background = chart.color;
      const name = document.createElement('span');
      name.textContent = `Chart ${chart.index}`;
      const error = document.createElement('span');
      error.className = 'chart-error';
      error.textContent = Number(chart.error).toPrecision(4);
      row.append(swatch, name, error);
      row.addEventListener('pointerenter', () => {
        chartRequest(`/chart-hover?chart=${chart.index}`).catch(console.error);
      });
      row.addEventListener('focus', () => {
        chartRequest(`/chart-hover?chart=${chart.index}`).catch(console.error);
      });
      row.addEventListener('click', async () => {
        await chartRequest(`/chart-select?chart=${chart.index}`);
        lastState = '';
        refreshCharts();
      });
      chartList.append(row);
    }
  }
  async function refreshCharts() {
    try { renderCharts(await chartRequest('/chart-state')); }
    catch (error) { console.error(error); }
  }
  chartList.addEventListener('pointerleave', () => {
    chartRequest('/chart-hover?chart=-1').catch(console.error);
  });
  document.getElementById('chart-picker').addEventListener('focusout', event => {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      chartRequest('/chart-hover?chart=-1').catch(console.error);
    }
  });
  refreshCharts();
  setInterval(refreshCharts, 300);
</script>
"""


def _chart_colors(count: int) -> np.ndarray:
    """Produce stable, vivid colors for any number of charts."""
    colors = [colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.78, 0.98)
              for index in range(count)]
    return np.rint(np.asarray(colors) * 255).astype(np.uint8)


def _numpy_points(points: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    result = np.asarray(points, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError("x_point_cloud must have shape (N, 3) with N > 0.")
    if not np.isfinite(result).all():
        raise ValueError("x_point_cloud must contain only finite values.")
    return result


def _model_tensor(model: torch.nn.Module, values: np.ndarray) -> torch.Tensor:
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    return torch.as_tensor(
        values,
        dtype=reference.dtype if reference is not None else torch.float32,
        device=reference.device if reference is not None else None,
    )


def _encode_decode(model: torch.nn.Module, x: torch.Tensor, chart: int):
    """Run one chart and check the batch shapes expected by the viewer."""
    z = model.encode(x, chart)
    if z.shape != (len(x), 2):
        raise ValueError(f"CAE.encode(x, {chart}) must return (B, 2); got {tuple(z.shape)}.")
    recon = model.decode(z, chart)
    if recon.shape != x.shape:
        raise ValueError(
            f"CAE.decode(z, {chart}) must return (B, 3); got {tuple(recon.shape)}."
        )
    return z, recon


def _chart_data(model: torch.nn.Module, points: np.ndarray, batch_size: int):
    """Cache each chart's aggregate posterior and reconstruction errors."""
    latent_chunks: list[list[np.ndarray]] = [[] for _ in range(model.chart_amount)]
    error_chunks: list[list[np.ndarray]] = [[] for _ in range(model.chart_amount)]
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            x = _model_tensor(model, points[start : start + batch_size])
            for chart in range(model.chart_amount):
                z, recon = _encode_decode(model, x, chart)
                error = torch.sum((recon - x) ** 2, dim=1)
                latent_chunks[chart].append(z.detach().cpu().numpy())
                error_chunks[chart].append(error.detach().cpu().numpy())
    latent = np.stack([np.concatenate(chunks) for chunks in latent_chunks]).astype(np.float32)
    errors = np.stack([np.concatenate(chunks) for chunks in error_chunks]).astype(np.float32)
    if not np.isfinite(latent).all() or not np.isfinite(errors).all():
        raise ValueError("CAE returned non-finite latent coordinates or reconstruction errors.")
    return latent, errors


@dataclass
class SplitView:
    """Running scene servers and wrapper returned by :func:`open_split_view`.

    Each server can be stopped independently. ``stop()`` stops all three;
    ``close()`` is an alias retained for existing callers. Stop calls are
    idempotent, so an individual server can be stopped before ``stop()``.
    """

    latent_server: viser.ViserServer
    x_server: viser.ViserServer
    split_view: ViserSplitView
    _latent_running: bool = field(default=True, init=False, repr=False)
    _x_running: bool = field(default=True, init=False, repr=False)
    _split_running: bool = field(default=True, init=False, repr=False)
    _stop_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def url(self) -> str:
        return self.split_view.url

    def stop_latent_server(self) -> None:
        """Stop the latent-chart Viser server."""
        with self._stop_lock:
            if self._latent_running:
                self.latent_server.stop()
                self._latent_running = False

    def stop_x_server(self) -> None:
        """Stop the observation-space Viser server."""
        with self._stop_lock:
            if self._x_running:
                self.x_server.stop()
                self._x_running = False

    def stop_split_server(self) -> None:
        """Stop the split-page HTTP server."""
        with self._stop_lock:
            if self._split_running:
                self.split_view.close()
                self._split_running = False

    def stop(self) -> None:
        """Stop the split page and both Viser servers."""
        try:
            self.stop_split_server()
        finally:
            try:
                self.stop_latent_server()
            finally:
                self.stop_x_server()

    def close(self) -> None:
        """Alias for :meth:`stop`."""
        self.stop()


def open_split_view(
    x_point_cloud: np.ndarray | torch.Tensor,
    cae: torch.nn.Module,
    *,
    epsilon: float | None = None,
    batch_size: int = 1024,
    host: str = "127.0.0.1",
    latent_port: int = 8080,
    x_port: int = 8081,
    split_port: int = 8082,
    open_browser: bool = True,
    point_size: float = 0.035,
    observation_opacity: float = 0.65,
) -> SplitView:
    """Open a split view of one CAE chart and a three-dimensional point cloud.

    The latent marker is draggable. The x-space marker shows its decoded point
    and cannot be dragged. The chart list is sorted by the current point's
    squared Euclidean reconstruction error under every chart. Selecting a chart
    encodes the current decoded x-space point into it, then decodes that latent
    coordinate to update the x-space marker. The latent cloud for a chart
    contains the encodings of input points whose reconstruction error on that
    chart is at most ``epsilon``. ``observation_opacity`` controls the alpha
    of the colored x-space cloud. Precomputation may take time for large clouds.
    """
    points = _numpy_points(x_point_cloud)
    if not isinstance(cae, torch.nn.Module):
        raise TypeError("cae must be a torch.nn.Module with encode and decode methods.")
    if getattr(cae, "x_dim", None) != 3 or getattr(cae, "z_dim", None) != 2:
        raise ValueError("cae must have x_dim == 3 and z_dim == 2.")
    if not isinstance(getattr(cae, "chart_amount", None), int) or cae.chart_amount < 1:
        raise ValueError("cae.chart_amount must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if point_size <= 0:
        raise ValueError("point_size must be positive.")
    if not np.isfinite(observation_opacity) or not 0 < observation_opacity <= 1:
        raise ValueError("observation_opacity must be in (0, 1].")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be finite and nonnegative.")

    latent, errors = _chart_data(cae, points, batch_size)
    best_errors = errors.min(axis=0)
    best_chart = np.argmin(errors, axis=0)
    chart_colors = _chart_colors(cae.chart_amount)
    if epsilon is None:
        epsilon = float(np.quantile(best_errors, 0.9))
    epsilon_max = max(float(errors.max()), float(epsilon), 1e-6)
    # For an initial point, choose a sample near the median quality, rather
    # than an extreme or an arbitrary first row.
    seed_index = int(np.argmin(np.abs(best_errors - np.median(best_errors))))
    x_seed = points[seed_index : seed_index + 1]

    def rank(x_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = _model_tensor(cae, x_point.reshape(1, 3))
            scores = np.empty(cae.chart_amount, dtype=np.float64)
            encodings = np.empty((cae.chart_amount, 2), dtype=np.float32)
            for chart in range(cae.chart_amount):
                z, recon = _encode_decode(cae, x, chart)
                scores[chart] = torch.sum((recon - x) ** 2).item()
                encodings[chart] = z[0].detach().cpu().numpy()
        if not np.isfinite(scores).all() or not np.isfinite(encodings).all():
            raise ValueError("CAE returned non-finite coordinates or reconstruction errors.")
        return scores, encodings

    initial_scores, initial_encodings = rank(x_seed[0])
    active_chart = int(np.argmin(initial_scores))
    active_z = initial_encodings[active_chart].copy()

    def decode(z: np.ndarray, chart: int) -> np.ndarray:
        with torch.no_grad():
            z_tensor = _model_tensor(cae, z.reshape(1, 2))
            reconstructed = cae.decode(z_tensor, chart)
            if reconstructed.shape != (1, 3):
                raise ValueError("CAE.decode must return shape (1, 3).")
            result = reconstructed[0].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("CAE.decode returned non-finite coordinates.")
        return result

    active_x = decode(active_z, active_chart)
    initial_scores, _ = rank(active_x)
    current_scores = initial_scores
    lock = threading.RLock()

    latent_server = viser.ViserServer(host=host, port=latent_port, label="CAE latent chart")
    split_view: ViserSplitView | None = None
    try:
        x_server = viser.ViserServer(host=host, port=x_port, label="X space")
    except Exception:
        latent_server.stop()
        raise

    try:
        latent_server.scene.set_up_direction("+z")
        x_server.scene.set_up_direction("+z")
        latent_extent = max(float(np.max(np.abs(latent))), 1.0)
        x_extent = max(float(np.max(np.abs(points))), 1.0)
        latent_server.scene.add_grid(
            "/latent/grid", width=2 * latent_extent, height=2 * latent_extent,
            plane="xy", cell_size=latent_extent / 6, section_size=latent_extent / 3,
        )
        latent_server.scene.add_label(
            "/latent/x_label", "latent x", position=(latent_extent, 0, 0.02)
        )
        latent_server.scene.add_label(
            "/latent/y_label", "latent y", position=(0, latent_extent, 0.02)
        )
        latent_server.initial_camera.position = (0, 0, max(3 * latent_extent, 3))
        latent_server.initial_camera.look_at = (0, 0, 0)
        latent_server.initial_camera.up = (0, 1, 0)

        x_server.scene.add_grid(
            "/x/grid", width=2 * x_extent, height=2 * x_extent, plane="xy",
            cell_size=x_extent / 6, section_size=x_extent / 3,
        )
        # Viser point clouds have RGB colors but no opacity control. Small
        # Gaussian splats give the observation cloud real alpha blending.
        sigma = max(point_size * 0.5, x_extent * 0.003)
        covariance = np.broadcast_to(
            np.eye(3, dtype=np.float32) * sigma**2,
            (len(points), 3, 3),
        ).copy()
        x_server.scene.add_gaussian_splats(
            "/x/points", centers=points, covariances=covariance,
            rgbs=chart_colors[best_chart].astype(np.float32) / 255.0,
            opacities=np.full((len(points), 1), observation_opacity, dtype=np.float32),
        )
        highlight = x_server.scene.add_point_cloud(
            "/x/highlight", points=points[:1],
            colors=tuple(int(channel) for channel in chart_colors[active_chart]),
            point_size=point_size * 1.8, point_shape="circle",
            point_shading="flat", visible=False,
        )
        x_server.scene.add_frame("/x/axes", axes_length=x_extent / 3, axes_radius=0.01)
        x_server.initial_camera.position = (2.3 * x_extent, -3 * x_extent, 2.3 * x_extent)
        x_server.initial_camera.look_at = (0, 0, 0)
        x_server.initial_camera.up = (0, 0, 1)

        latent_xyz = np.column_stack((latent[active_chart], np.zeros(len(points)))).astype(np.float32)
        accepted = errors[active_chart] <= epsilon
        cloud = latent_server.scene.add_point_cloud(
            "/latent/posterior", points=latent_xyz[accepted] if accepted.any() else latent_xyz[:1],
            colors=chart_colors[best_chart[accepted]] if accepted.any() else chart_colors[best_chart[:1]],
            point_size=point_size,
            point_shape="circle", point_shading="flat", visible=bool(accepted.any()),
        )
        latent_marker = latent_server.scene.add_icosphere(
            "/latent/current", radius=0.04 * latent_extent,
            color=tuple(int(channel) for channel in chart_colors[int(np.argmin(initial_scores))]),
            subdivisions=2,
            position=(float(active_z[0]), float(active_z[1]), 0.04 * latent_extent),
        )
        x_marker = x_server.scene.add_icosphere(
            "/x/current", radius=0.04 * x_extent,
            color=tuple(int(channel) for channel in chart_colors[int(np.argmin(initial_scores))]),
            subdivisions=2,
            position=tuple(active_x),
        )

        chart_panel_url = f"http://{host}:{split_port}/chart-panel"
        panel_height = min(420, 90 + 31 * cae.chart_amount)
        latent_server.gui.add_html(
            f'<iframe title="Charts" src="{chart_panel_url}" '
            f'style="display:block;width:100%;height:{panel_height}px;border:0"></iframe>'
        )
        epsilon_slider = latent_server.gui.add_slider(
            "Max squared reconstruction error", min=0.0, max=epsilon_max,
            step=max(epsilon_max / 1000, 1e-8), initial_value=float(epsilon),
        )
        latent_server.gui.add_markdown("Drag the colored point to explore this chart. Hover or click a chart in the list above.")
        coverage = latent_server.gui.add_markdown("")
        coordinates = latent_server.gui.add_markdown("")
        x_readout = x_server.gui.add_markdown("")

        def refresh_cloud() -> None:
            xyz = np.column_stack((latent[active_chart], np.zeros(len(points)))).astype(np.float32)
            mask = errors[active_chart] <= epsilon_slider.value
            cloud.points = xyz[mask] if mask.any() else xyz[:1]
            cloud.colors = chart_colors[best_chart[mask]] if mask.any() else chart_colors[best_chart[:1]]
            cloud.visible = bool(mask.any())
            coverage.content = (
                f"**Chart {active_chart}:** {int(mask.sum())} / {len(points)} input points "
                f"reconstruct within squared error <= {epsilon_slider.value:.5g}."
            )

        def refresh_view() -> None:
            nonlocal current_scores
            scores, _ = rank(active_x)
            current_scores = scores
            latent_marker.position = (float(active_z[0]), float(active_z[1]), 0.04 * latent_extent)
            marker_color = tuple(int(channel) for channel in chart_colors[int(np.argmin(scores))])
            latent_marker.color = marker_color
            x_marker.position = tuple(active_x)
            x_marker.color = marker_color
            coordinates.content = f"**Latent:** [{active_z[0]:.4g}, {active_z[1]:.4g}]"
            x_readout.content = (
                f"**Decoded x:** [{active_x[0]:.4g}, {active_x[1]:.4g}, {active_x[2]:.4g}]  \n"
                f"**Active chart {active_chart} error:** {scores[active_chart]:.5g}"
            )
            refresh_cloud()

        @latent_marker.on_drag
        async def _on_latent_drag(event: viser.SceneNodeDragEvent) -> None:
            if event.phase in ("update", "end"):
                nonlocal active_z, active_x
                with lock:
                    active_z = np.asarray(event.end_position[:2], dtype=np.float32).copy()
                    active_x = decode(active_z, active_chart)
                    refresh_view()

        def select_chart(new_chart: int) -> None:
            nonlocal active_chart, active_z, active_x
            with lock:
                if new_chart == active_chart:
                    return
                # Keep the current decoded point as the transition anchor.
                _, encodings = rank(active_x)
                active_chart = new_chart
                active_z = encodings[new_chart].copy()
                active_x = decode(active_z, active_chart)
                refresh_view()

        def set_highlight(chart: int | None) -> None:
            with lock:
                if chart is None:
                    highlight.visible = False
                    return
                selected_points = points[best_chart == chart]
                highlight.points = selected_points if len(selected_points) else points[:1]
                highlight.colors = chart_colors[chart].copy()
                highlight.visible = bool(len(selected_points))

        def on_request(path: str) -> tuple[str, bytes] | None:
            parsed = urlparse(path)
            if parsed.path == "/chart-panel":
                page = (
                    '<!doctype html><html><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'</head><body>{_CHART_CONTROLS_HTML}</body></html>'
                )
                return "text/html; charset=utf-8", page.encode("utf-8")
            if parsed.path not in ("/chart-state", "/chart-hover", "/chart-select"):
                return None
            with lock:
                if parsed.path == "/chart-state":
                    order = np.argsort(current_scores, kind="stable")
                    data = {
                        "active": active_chart,
                        "charts": [
                            {
                                "index": int(chart),
                                "error": float(current_scores[chart]),
                                "color": "#" + "".join(f"{int(value):02x}" for value in chart_colors[chart]),
                            }
                            for chart in order
                        ],
                    }
                else:
                    try:
                        chart = int(parse_qs(parsed.query)["chart"][0])
                    except (KeyError, ValueError, IndexError):
                        return None
                    if parsed.path == "/chart-hover" and chart == -1:
                        set_highlight(None)
                    elif 0 <= chart < cae.chart_amount:
                        if parsed.path == "/chart-hover":
                            set_highlight(chart)
                        else:
                            select_chart(chart)
                    else:
                        return None
                    data = {"ok": True}
            return "application/json; charset=utf-8", json.dumps(data).encode("utf-8")

        @epsilon_slider.on_update
        def _on_epsilon_change(_event) -> None:
            with lock:
                refresh_cloud()

        refresh_view()
        x_server.gui.main_panel.minimize()
        split_view = ViserSplitView(
            [
                SceneView("CAE latent chart", f"http://{host}:{latent_port}/"),
                SceneView("X space", f"http://{host}:{x_port}/"),
            ],
            host=host,
            port=split_port,
            on_request=on_request,
        )
        split_view.start(open_browser=open_browser)
    except Exception:
        if split_view is not None:
            split_view.close()
        x_server.stop()
        latent_server.stop()
        raise
    return SplitView(latent_server, x_server, split_view)
