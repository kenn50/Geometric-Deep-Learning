# %% [markdown]
# # Sinkhorn: decoding a uniform 3D cube into a torus
#
# Run the cells in order in VS Code's Interactive Window. The decoder is a
# trainable map R^3 -> R^3; the latent samples are not optimized. Fresh uniform
# cube samples are used for training, and the same held-out latent points are
# followed throughout the animation. Only Sinkhorn divergence trains the model.
#
# Both point clouds have three coordinates. The torus target is a 2D surface
# embedded in 3D, so this experiment does not impose an invertible chart map.
#
# Training records snapshots. The final browser plot has Play/Pause controls,
# a training-step slider, and a loss curve; there are no live notebook callbacks.
# Interrupting training still leaves the recorded history available.
#
# References:
# - https://proceedings.mlr.press/v89/feydy19a.html
# - https://www.kernel-operations.io/geomloss/api/pytorch-api.html

# %% Imports
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import nn
from geomloss import SamplesLoss
from tqdm.auto import trange


# %% Settings -- edit these before running the experiment
@dataclass
class Settings:
    seed: int = 7
    steps: int = 1300
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_width: int = 128
    # GeomLoss uses cost = ||x-y||^2 / 2, with epsilon = blur**2.
    blur: float = 0.05
    scaling: float = 0.8
    target_points: int = 4096
    display_points: int = 4096
    probe_points: int = 512
    record_every: int = 40
    major_radius: float = 1.0
    minor_radius: float = 0.35
    cube_half_width: float = 1.0


# %% Sampling and decoder
def sample_cube(count, half_width=1.0, *, device="cpu", generator=None):
    return half_width * (
        2 * torch.rand(count, 3, device=device, generator=generator) - 1
    )


def sample_torus(count, major_radius=1.0, minor_radius=0.35, *, device="cpu", generator=None):
    """Sample uniformly in surface area, using rejection sampling of the tube angle."""
    if not 0 < minor_radius < major_radius:
        raise ValueError("Require 0 < minor_radius < major_radius.")
    angles = []
    remaining = count
    while remaining:
        candidates = 2 * math.pi * torch.rand(
            max(2 * remaining, 32), device=device, generator=generator
        )
        acceptance = (major_radius + minor_radius * candidates.cos()) / (
            major_radius + minor_radius
        )
        accepted = candidates[
            torch.rand(candidates.shape, device=device, generator=generator) < acceptance
        ][:remaining]
        angles.append(accepted)
        remaining -= len(accepted)
    v = torch.cat(angles)
    u = 2 * math.pi * torch.rand(count, device=device, generator=generator)
    radial = major_radius + minor_radius * v.cos()
    return torch.stack(
        (radial * u.cos(), radial * u.sin(), minor_radius * v.sin()), dim=1
    )


class Decoder(nn.Module):
    """Plain residual MLP initialized close to the identity map."""

    def __init__(self, width=128):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 3),
        )
        # Start close to the cube, while allowing gradients into every layer
        # on the first optimizer step (a zero final weight blocks those).
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z):
        return z + self.residual(z)


# %% Training and snapshot recording
@dataclass
class Snapshot:
    step: int
    seconds: float
    loss: float
    points: np.ndarray


@dataclass
class Experiment:
    decoder: Decoder
    latent: np.ndarray
    target: np.ndarray
    history: list[Snapshot]
    settings: Settings
    device: str


def train_experiment(settings=None, *, device=None, progress=True):
    settings = settings or Settings()
    counts = (settings.batch_size, settings.target_points, settings.display_points,
              settings.probe_points, settings.record_every, settings.hidden_width)
    if settings.steps < 0 or any(value < 1 for value in counts):
        raise ValueError("steps must be nonnegative; point counts and record_every must be positive.")
    if settings.cube_half_width <= 0:
        raise ValueError("cube_half_width must be positive.")
    if not math.isfinite(settings.blur) or settings.blur <= 0:
        raise ValueError("blur must be positive and finite.")
    if not 0 < settings.scaling < 1:
        raise ValueError("scaling must be between zero and one.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(settings.seed)
    train_rng = torch.Generator(device=device).manual_seed(settings.seed + 1)
    display_rng = torch.Generator(device=device).manual_seed(settings.seed + 2)
    probe_rng = torch.Generator(device=device).manual_seed(settings.seed + 3)

    decoder = Decoder(settings.hidden_width).to(device)
    criterion = SamplesLoss(
        loss="sinkhorn", p=2, blur=settings.blur, scaling=settings.scaling,
        debias=True, backend="tensorized",
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=settings.learning_rate)
    target = sample_torus(
        settings.target_points, settings.major_radius, settings.minor_radius,
        device=device, generator=train_rng,
    )
    latent = sample_cube(settings.display_points, settings.cube_half_width,
                         device=device, generator=display_rng)
    probe_z = sample_cube(settings.probe_points, settings.cube_half_width,
                          device=device, generator=probe_rng)
    probe_target = sample_torus(
        settings.probe_points, settings.major_radius, settings.minor_radius,
        device=device, generator=probe_rng,
    )
    history = []
    started = perf_counter()

    @torch.no_grad()
    def record(step):
        probe_output = decoder(probe_z)
        output = decoder(latent).detach().cpu().numpy().copy()
        value = criterion(probe_output, probe_target).detach().item()
        if not math.isfinite(value) or not np.isfinite(output).all():
            raise FloatingPointError("Non-finite loss or decoded points; reduce learning_rate.")
        history.append(Snapshot(step, perf_counter() - started, value, output))

    record(0)
    last_step = 0
    steps = trange(1, settings.steps + 1, desc="Sinkhorn decoder", disable=not progress)
    try:
        for step in steps:
            z = sample_cube(settings.batch_size, settings.cube_half_width,
                            device=device, generator=train_rng)
            indices = torch.randint(len(target), (settings.batch_size,),
                                    device=device, generator=train_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(decoder(z), target[indices])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss; reduce learning_rate.")
            loss.backward()
            optimizer.step()
            last_step = step
            if step % settings.record_every == 0 or step == settings.steps:
                record(step)
                steps.set_postfix(probe_loss=f"{history[-1].loss:.5f}")
    except KeyboardInterrupt:
        print("Training interrupted; keeping the model and recorded snapshots.")
    finally:
        steps.close()

    if history[-1].step != last_step:
        record(last_step)
    return Experiment(
        decoder, latent.cpu().numpy(), target.cpu().numpy(), history,
        settings, str(device),
    )


# %% Interactive browser visualization
def make_history_plot(experiment):
    history = experiment.history
    if not history:
        raise ValueError("No snapshots were recorded.")
    latent, target = experiment.latent, experiment.target
    figure = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "scene"}] * 3,
               [{"type": "xy", "colspan": 3}, None, None]],
        row_heights=[0.78, 0.22], vertical_spacing=0.12,
        subplot_titles=["Uniform latent cube (fixed)", "Decoder output", "Target torus",
                        "Sinkhorn divergence on fixed held-out probe points"],
    )

    # Trace 0: cube wireframe. Trace 1: latent points. Trace 2: decoder output.
    half = experiment.settings.cube_half_width
    corners = np.array([[x, y, z] for x in (-half, half)
                        for y in (-half, half) for z in (-half, half)])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(corners[a] != corners[b]) == 1]
    lines = np.full((len(edges), 3, 3), np.nan)
    lines[:, 0] = corners[[a for a, _ in edges]]
    lines[:, 1] = corners[[b for _, b in edges]]
    xyz = lines.reshape(-1, 3)
    figure.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
        line=dict(color="black", width=3), hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    for column, values, name in ((1, latent, "Latent"), (2, history[0].points, "Decoded")):
        figure.add_trace(go.Scatter3d(
            x=values[:, 0], y=values[:, 1], z=values[:, 2], mode="markers", name=name,
            marker=dict(size=2, color="#222222"),
            customdata=np.arange(len(latent)),
            hovertemplate="Point %{customdata}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            showlegend=False,
        ), row=1, col=column)

    # Trace 3: target. Thin the display only; training keeps the full target pool.
    shown_target = target[:min(len(target), max(1000, len(latent)))]
    figure.add_trace(go.Scatter3d(
        x=shown_target[:, 0], y=shown_target[:, 1], z=shown_target[:, 2],
        mode="markers", marker=dict(size=2, color="#555555"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=3)

    # Traces 4 and 5: complete loss curve and current animation position.
    figure.add_trace(go.Scatter(
        x=[item.step for item in history], y=[item.loss for item in history],
        mode="lines", line=dict(color="#3465a4"), showlegend=False,
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=[history[0].step], y=[history[0].loss], mode="markers",
        marker=dict(color="red", size=10), showlegend=False,
    ), row=2, col=1)

    frames = []
    for item in history:
        frames.append(go.Frame(
            name=str(item.step), traces=[2, 5],
            data=[go.Scatter3d(x=item.points[:, 0], y=item.points[:, 1], z=item.points[:, 2]),
                  go.Scatter(x=[item.step], y=[item.loss])],
            layout=go.Layout(title=dict(text=(
                f"Cube → torus | step {item.step} | Sinkhorn {item.loss:.5f} | "
                f"training time {item.seconds:.1f} s"
            ))),
        ))
    figure.frames = frames

    bound = 1.2 * max(half, experiment.settings.major_radius + experiment.settings.minor_radius,
                      max(float(np.abs(item.points).max()) for item in history))
    figure.update_scenes(
        aspectmode="cube", dragmode="orbit", uirevision="sinkhorn-camera",
        xaxis=dict(range=[-bound, bound], title="x"),
        yaxis=dict(range=[-bound, bound], title="y"),
        zaxis=dict(range=[-bound, bound], title="z"),
        camera=dict(eye=dict(x=1.6, y=-2, z=1.2)),
    )
    figure.update_xaxes(title_text="Training step", row=2, col=1)
    figure.update_yaxes(title_text="Sinkhorn divergence", row=2, col=1)
    figure.update_layout(
        title=frames[0].layout.title,
        height=850, margin=dict(l=50, r=40, t=100, b=130),
        uirevision="sinkhorn-camera",
        sliders=[dict(
            active=0, x=0.13, len=0.87, y=-0.05,
            currentvalue=dict(prefix="Training step: "),
            steps=[dict(
                method="animate", label=str(item.step),
                args=[[str(item.step)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                            transition=dict(duration=0))],
            ) for item in history],
        )],
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.05, showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(
                    fromcurrent=True, mode="immediate", frame=dict(duration=180, redraw=True),
                    transition=dict(duration=0),
                )]),
                dict(label="Pause", method="animate", args=[[None], dict(
                    mode="immediate", frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0),
                )]),
            ],
        )],
    )
    return figure


# %% Run training -- change settings here, then rerun this cell
if __name__ == "__main__":
    settings = Settings()  # Uses the editable settings near the top.
    experiment = train_experiment(settings)
    decoder = experiment.decoder
    history = experiment.history
    print(f"Device: {experiment.device}")
    print(f"Fixed-probe Sinkhorn: {history[0].loss:.5f} → {history[-1].loss:.5f}")


# %% Replay training in the browser -- no Panel, server, or periodic callbacks
if __name__ == "__main__":
    fig = make_history_plot(experiment)
    # Save a standalone copy with Plotly included, requiring no internet/server.
    if "__file__" in globals():
        root = Path(__file__).resolve().parents[1]
    else:
        root = next((p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "pyproject.toml").exists()), Path.cwd())
    html_path = root / "reports" / "sinkhorn.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=True, auto_play=False)
    print(f"Saved interactive history: {html_path}")
    fig.show(renderer="browser", auto_play=False)

# %%
