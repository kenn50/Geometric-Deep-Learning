# Created by GPT 6.1 Sol
# %% [markdown]
# # Sinkhorn with eight learned surface patches and inverse encoders
#
# Eight independent decoders map uniform 2D squares into R^3. Each update draws
# 128 fresh latent points per chart, concatenates all 1024 decoded points, and
# compares that combined cloud to 1024 torus points with one GeomLoss Sinkhorn
# divergence. Each decoder is D_i(z) = c_i + S*f_i(z), with one positive
# learnable S shared by all charts. A Jacobian penalty pushes both singular
# values of the unscaled f_i toward 1. There are no per-patch target assignments.
# Each chart also has a 3D -> 2D encoder trained with E_i(D_i(z)) ~= z.
# Decoded training points are detached for this loss so encoders learn the
# current charts without changing the decoder's generative objective.
# After half the training updates, real torus points receive a reconstruction
# loss: each point uses the
# chart with the smallest mean squared D_i(E_i(x)) error, training both networks.
#
# Run the cells in order. The last cell opens a browser replay with eight latent
# squares, all decoded patches, the target torus, and a fixed-probe loss curve.
# Each chart has a consistent color in latent and decoded space; the legend
# can hide individual patches.

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

# Reuse the surface-area-uniform torus sampler from the original experiment.
try:
    from experiments.vibe.sinkhorn import sample_torus
except ModuleNotFoundError as error:
    if error.name not in ("experiments", "experiments.vibe"):
        raise
    from sinkhorn import sample_torus


# %% Settings
N_PATCHES = 8
PATCH_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#7A5195", "#555555",
)


@dataclass
class Settings:
    seed: int = 7
    steps: int = 2000
    points_per_patch: int = 128  # 8 * 128 = 1024 decoded points per update.
    learning_rate: float = 1e-3
    jacobian_weight: float = 1.0  # Weight of mean((singular_values(J_f) - 1)**2); 0 disables it.
    encoder_weight: float = 1.0  # Weight of latent recovery MSE; trains encoders only.
    reconstruction_weight: float = 1.0  # Best-chart reconstruction weight; active in second half only.
    initial_singular_scale: float = 1.0  # Initial shared forward multiplier S.
    hidden_width: int = 128
    blur: float = 0.05
    scaling: float = 0.8
    target_points: int = 4096
    display_points_per_patch: int = 4000
    probe_points_per_patch: int = 128
    record_every: int = 40
    major_radius: float = 1.0
    minor_radius: float = 0.35
    square_half_width: float = 0.5  # Latent squares span [-0.5, 0.5]^2.
    initial_patch_scale: float = 1.0  # Unscaled f starts with near-unit singular values.
    initial_center_scale: float = 0.45


# %% Eight independent 2D -> 3D decoders
def sample_latents(count_per_patch, half_width=1.0, *, device="cpu", generator=None):
    return half_width * (
        2 * torch.rand(N_PATCHES, count_per_patch, 2, device=device, generator=generator) - 1
    )


class PatchDecoder(nn.Module):
    """D_i(z) = center + shared_scale * f_i(z), with an unscaled shape MLP."""

    def __init__(self, center, width=128, initial_scale=1.0):
        super().__init__()
        self.center = nn.Parameter(center.clone())
        self.affine = nn.Linear(2, 3, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 3),
        )
        # Different generic plane orientations break symmetry between patches.
        # This initialization contains no torus parameterization.
        basis, _ = torch.linalg.qr(torch.randn(3, 2), mode="reduced")
        with torch.no_grad():
            self.affine.weight.copy_(initial_scale * basis)
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def unscaled(self, z):
        """f_i(z), whose Jacobian singular values are regularized toward 1."""
        return self.affine(z) + self.residual(z)

    def forward(self, z, shared_scale):
        return self.center + shared_scale * self.unscaled(z)


class PatchDecoders(nn.Module):
    def __init__(self, width=128, initial_scale=1.0, center_scale=0.45,
                 initial_singular_scale=1.0):
        super().__init__()
        if not math.isfinite(initial_singular_scale) or initial_singular_scale <= 0:
            raise ValueError("initial_singular_scale must be finite and positive.")
        # A single scalar parameter belongs to the whole atlas, not each decoder.
        self.log_singular_scale = nn.Parameter(torch.tensor(math.log(initial_singular_scale)))
        # Eight separated starting locations at the vertices of a cube.
        centers = center_scale * torch.tensor(
            [[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]
        )
        self.decoders = nn.ModuleList([
            PatchDecoder(center, width, initial_scale) for center in centers
        ])

    @property
    def singular_scale(self):
        """Shared positive multiplier S used directly in every decoder output."""
        return self.log_singular_scale.exp()

    def unscaled(self, z):
        # z: (N_PATCHES, points_per_patch, 2) -> combined: (N_PATCHES * points_per_patch, 3).
        return torch.cat([
            decoder.unscaled(z[chart]) for chart, decoder in enumerate(self.decoders)
        ], dim=0)

    def scale_unscaled(self, unscaled):
        """Apply shared S around each chart's independently learned center."""
        patches = unscaled.reshape(N_PATCHES, -1, 3)
        centers = torch.stack([decoder.center for decoder in self.decoders])
        return (centers[:, None, :] + self.singular_scale * patches).reshape(-1, 3)

    def forward(self, z):
        return self.scale_unscaled(self.unscaled(z))


class PatchEncoder(nn.Module):
    """Plain 3D -> 2D MLP; unbounded output avoids saturation at square edges."""

    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, points):
        return self.network(points)


class PatchAutoencoders(PatchDecoders):
    """Eight decoder/encoder pairs with one shared decoder singular scale."""

    def __init__(self, width=128, initial_scale=1.0, center_scale=0.45,
                 initial_singular_scale=1.0):
        super().__init__(width, initial_scale, center_scale, initial_singular_scale)
        self.encoders = nn.ModuleList([PatchEncoder(width) for _ in range(N_PATCHES)])

    def encode(self, points, chart):
        """Encode points with a chosen chart's encoder: (..., 3) -> (..., 2)."""
        return self.encoders[chart](points)

    def decode(self, z, chart):
        """Decode latent coordinates with a chosen chart: (..., 2) -> (..., 3)."""
        return self.decoders[chart](z, self.singular_scale)

    def encode_patches(self, decoded):
        """Encode concatenated decoder output using its originating charts."""
        patches = decoded.reshape(N_PATCHES, -1, 3)
        return torch.stack([
            self.encode(points, chart) for chart, points in enumerate(patches)
        ])


def reconstruction_loss(model, points):
    """Best-chart reconstruction MSE on real data, differentiable in E and D."""
    errors = torch.stack([
        (model.decode(model.encode(points, chart), chart) - points).square().mean(dim=-1)
        for chart in range(N_PATCHES)
    ], dim=1)  # (points, charts).
    return errors.min(dim=1).values.mean()


def jacobian_isometry_loss(unscaled, z):
    """Mean squared deviation of unscaled shape Jacobian singular values from 1.

    Each decoder processes points independently, so differentiating a sum of
    output coordinates gives the corresponding Jacobian row at every point.
    create_graph=True allows the penalty to backpropagate into decoder weights.
    """
    rows = [
        torch.autograd.grad(
            unscaled[:, coordinate].sum(), z, create_graph=True,
        )[0]
        for coordinate in range(3)
    ]
    jacobian = torch.stack(rows, dim=-2)  # (N_PATCHES, points_per_patch, 3, 2).
    singular_values = torch.linalg.svdvals(jacobian)
    return (singular_values - 1.0).square().mean()


# %% Training -- combined Sinkhorn loss plus local Jacobian regularization
@dataclass
class Snapshot:
    step: int
    seconds: float
    loss: float
    points: np.ndarray  # (N_PATCHES, display_points_per_patch, 3).
    encoder_loss: float  # Fixed-probe latent recovery MSE.
    reconstruction_loss: float  # Fixed-probe best-chart real-data reconstruction MSE.
    singular_scale: float  # Learned shared S at this snapshot.


@dataclass
class Experiment:
    model: PatchAutoencoders
    latent: np.ndarray  # (N_PATCHES, display_points_per_patch, 2).
    target: np.ndarray
    history: list[Snapshot]
    settings: Settings
    device: str


def train_experiment(settings=None, *, device=None, progress=True):
    settings = settings or Settings()
    counts = (settings.points_per_patch, settings.display_points_per_patch,
              settings.probe_points_per_patch, settings.target_points,
              settings.hidden_width, settings.record_every)
    if settings.steps < 0 or any(value < 1 for value in counts):
        raise ValueError("steps must be nonnegative; point counts and record_every must be positive.")
    if settings.square_half_width <= 0 or settings.blur <= 0 or not 0 < settings.scaling < 1:
        raise ValueError("Require positive square_half_width and blur, and 0 < scaling < 1.")
    if not math.isfinite(settings.jacobian_weight) or settings.jacobian_weight < 0:
        raise ValueError("jacobian_weight must be finite and nonnegative.")
    if not math.isfinite(settings.encoder_weight) or settings.encoder_weight < 0:
        raise ValueError("encoder_weight must be finite and nonnegative.")
    if not math.isfinite(settings.reconstruction_weight) or settings.reconstruction_weight < 0:
        raise ValueError("reconstruction_weight must be finite and nonnegative.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(settings.seed)
    train_rng = torch.Generator(device=device).manual_seed(settings.seed + 1)
    display_rng = torch.Generator(device=device).manual_seed(settings.seed + 2)
    probe_rng = torch.Generator(device=device).manual_seed(settings.seed + 3)

    model = PatchAutoencoders(
        settings.hidden_width, settings.initial_patch_scale, settings.initial_center_scale,
        initial_singular_scale=settings.initial_singular_scale,
    ).to(device)
    criterion = SamplesLoss(
        loss="sinkhorn", p=2, blur=settings.blur, scaling=settings.scaling,
        debias=True, backend="tensorized",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    target = sample_torus(
        settings.target_points, settings.major_radius, settings.minor_radius,
        device=device, generator=train_rng,
    )
    latent = sample_latents(
        settings.display_points_per_patch, settings.square_half_width,
        device=device, generator=display_rng,
    )
    probe_z = sample_latents(
        settings.probe_points_per_patch, settings.square_half_width,
        device=device, generator=probe_rng,
    )
    probe_target = sample_torus(
        N_PATCHES * settings.probe_points_per_patch,
        settings.major_radius, settings.minor_radius,
        device=device, generator=probe_rng,
    )
    history = []
    started = perf_counter()

    @torch.no_grad()
    def record(step):
        output = model(latent).detach().reshape(N_PATCHES, -1, 3).cpu().numpy().copy()
        probe_output = model(probe_z)
        encoder_value = (model.encode_patches(probe_output) - probe_z).square().mean().item()
        reconstruction_value = reconstruction_loss(model, probe_target).item()
        scale_value = model.singular_scale.detach().item()
        value = criterion(probe_output, probe_target).detach().item()
        if (not all(math.isfinite(v) for v in (value, encoder_value, reconstruction_value, scale_value))
                or not np.isfinite(output).all()):
            raise FloatingPointError("Non-finite loss or decoded points; reduce learning_rate.")
        history.append(Snapshot(
            step, perf_counter() - started, value, output, encoder_value, reconstruction_value,
            scale_value,
        ))

    record(0)
    last_step = 0
    steps = trange(1, settings.steps + 1, desc="Sinkhorn patches + encoders", disable=not progress)
    try:
        for step in steps:
            z = sample_latents(
                settings.points_per_patch, settings.square_half_width,
                device=device, generator=train_rng,
            )
            z.requires_grad_(settings.jacobian_weight > 0)
            # Each decoder contributes points_per_patch points to the combined cloud.
            unscaled = model.unscaled(z)
            decoded = model.scale_unscaled(unscaled)
            indices = torch.randint(len(target), (len(decoded),), device=device, generator=train_rng)
            optimizer.zero_grad(set_to_none=True)
            real_points = target[indices]
            sinkhorn_loss = criterion(decoded, real_points)
            jacobian_loss = (
                jacobian_isometry_loss(unscaled, z)
                if settings.jacobian_weight > 0 else decoded.new_zeros(())
            )
            # Each encoder learns the inverse of its own current decoder.
            # Detach both sides: this term updates encoder parameters only.
            encoder_loss = (
                (model.encode_patches(decoded.detach()) - z.detach()).square().mean()
                if settings.encoder_weight > 0 else decoded.new_zeros(())
            )
            reconstruction_active = step > settings.steps // 2
            loss = (
                sinkhorn_loss + settings.encoder_weight * encoder_loss
                if reconstruction_active
                else (sinkhorn_loss + settings.jacobian_weight * jacobian_loss
                    + settings.encoder_weight * encoder_loss)
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss; reduce learning_rate.")
            loss.backward()
            optimizer.step()
            last_step = step
            if step % settings.record_every == 0 or step == settings.steps:
                record(step)
                steps.set_postfix(
                    probe_loss=f"{history[-1].loss:.5f}",
                    jacobian_loss=f"{jacobian_loss.detach().item():.5f}",
                    encoder_mse=f"{history[-1].encoder_loss:.5f}",
                    reconstruction_mse=f"{history[-1].reconstruction_loss:.5f}",
                    reconstruction_active=reconstruction_active,
                    S=f"{model.singular_scale.detach().item():.5f}",
                )
    except KeyboardInterrupt:
        print("Training interrupted; keeping recorded snapshots and the model.")
    finally:
        steps.close()
    if history[-1].step != last_step:
        record(last_step)
    return Experiment(
        model, latent.cpu().numpy(), target.cpu().numpy(), history, settings, str(device)
    )


# %% Interactive replay
def make_history_plot(experiment):
    history, settings = experiment.history, experiment.settings
    if not history:
        raise ValueError("No snapshots were recorded.")
    figure = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}],
               [{"type": "xy", "colspan": 3}, None, None],
               [{"type": "xy", "colspan": 3}, None, None]],
        row_heights=[0.66, 0.17, 0.17], vertical_spacing=0.09,
        subplot_titles=[f"{N_PATCHES} 2D latent squares", "Combined decoder output", "Target torus",
                        "Sinkhorn divergence on fixed held-out probe points",
                        "Learned singular scale S (shared by all charts)"],
    )
    half = settings.square_half_width
    spacing = 1.3 * half
    columns = 4
    rows = math.ceil(N_PATCHES / columns)
    offsets = np.array([
        [(2 * (chart % columns) - (columns - 1)) * spacing,
         ((rows - 1) - 2 * (chart // columns)) * spacing]
        for chart in range(N_PATCHES)
    ])

    # First N_PATCHES traces: latent charts, shifted apart only for display.
    for chart, (u, v) in enumerate(offsets):
        z = experiment.latent[chart]
        figure.add_trace(go.Scattergl(
            x=z[:, 0] + u, y=z[:, 1] + v, mode="markers",
            name=f"Patch {chart + 1}", legendgroup=f"patch-{chart}", showlegend=False,
            marker=dict(size=2, color=PATCH_COLORS[chart]), customdata=z,
            hovertemplate=f"Patch {chart + 1}<br>u=%{{customdata[0]:.3f}}<br>v=%{{customdata[1]:.3f}}<extra></extra>",
        ), row=1, col=1)
        figure.add_shape(
            type="rect", x0=u-half, x1=u+half, y0=v-half, y1=v+half,
            line=dict(color=PATCH_COLORS[chart], width=1), row=1, col=1,
        )
        figure.add_annotation(x=u, y=v+1.13*half, text=f"Patch {chart + 1}",
                              font=dict(color=PATCH_COLORS[chart]),
                              showarrow=False, row=1, col=1)

    # Next N_PATCHES traces: decoded charts. Legend clicks can toggle each patch.
    for chart, points in enumerate(history[0].points):
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=2, color=PATCH_COLORS[chart]), name=f"Patch {chart + 1}",
            legendgroup=f"patch-{chart}", showlegend=True,
            hovertemplate=f"Patch {chart + 1}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})<extra></extra>",
        ), row=1, col=2)

    # Target, displayed with the same marker size as generated points.
    target = experiment.target[:N_PATCHES * settings.display_points_per_patch]
    figure.add_trace(go.Scatter3d(
        x=target[:, 0], y=target[:, 1], z=target[:, 2], mode="markers",
        marker=dict(size=2, color="#555555"), hoverinfo="skip", showlegend=False,
    ), row=1, col=3)
    # Loss history and the current replay position follow the target trace.
    figure.add_trace(go.Scatter(
        x=[s.step for s in history], y=[s.loss for s in history], mode="lines",
        line=dict(color="#3465a4"), showlegend=False,
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=[history[0].step], y=[history[0].loss], mode="markers",
        marker=dict(color="red", size=10), showlegend=False,
    ), row=2, col=1)
    # Shared scale history and its replay cursor have separate axes from the loss.
    figure.add_trace(go.Scatter(
        x=[s.step for s in history], y=[s.singular_scale for s in history], mode="lines",
        line=dict(color="#009E73"), name="Shared S", showlegend=False,
    ), row=3, col=1)
    figure.add_trace(go.Scatter(
        x=[history[0].step], y=[history[0].singular_scale], mode="markers",
        marker=dict(color="red", size=10), showlegend=False,
    ), row=3, col=1)

    frames = []
    for snapshot in history:
        frames.append(go.Frame(
            name=str(snapshot.step),
            traces=[*range(N_PATCHES, 2 * N_PATCHES),
                    2 * N_PATCHES + 2, 2 * N_PATCHES + 4],
            data=[go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]) for p in snapshot.points]
                 + [go.Scatter(x=[snapshot.step], y=[snapshot.loss]),
                    go.Scatter(x=[snapshot.step], y=[snapshot.singular_scale])],
            layout=go.Layout(title=dict(text=(
                f"{N_PATCHES} 2D patches -> torus | step {snapshot.step} | "
                f"Sinkhorn {snapshot.loss:.5f} | encoder MSE {snapshot.encoder_loss:.5f} | "
                f"reconstruction MSE {snapshot.reconstruction_loss:.5f} | "
                f"S {snapshot.singular_scale:.4f} | "
                f"{snapshot.seconds:.1f} s"
            ))),
        ))
    figure.frames = frames
    bound = 1.2 * max(
        settings.major_radius + settings.minor_radius,
        max(float(np.abs(snapshot.points).max()) for snapshot in history),
    )
    figure.update_scenes(
        aspectmode="cube", dragmode="orbit", uirevision="patch-camera",
        xaxis=dict(range=[-bound, bound], title="x"),
        yaxis=dict(range=[-bound, bound], title="y"),
        zaxis=dict(range=[-bound, bound], title="z"),
        camera=dict(eye=dict(x=1.6, y=-2, z=1.2)),
    )
    figure.update_xaxes(range=[-columns*spacing, columns*spacing], showticklabels=False, row=1, col=1)
    figure.update_yaxes(range=[-rows*spacing, rows*spacing], showticklabels=False,
                        scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_xaxes(title_text="Training step", row=2, col=1)
    figure.update_yaxes(title_text="Sinkhorn divergence", row=2, col=1)
    figure.update_xaxes(title_text="Training step", row=3, col=1)
    figure.update_yaxes(title_text="Shared S", rangemode="tozero", row=3, col=1)
    figure.update_layout(
        title=frames[0].layout.title, height=1050,
        margin=dict(l=50, r=40, t=120, b=130), uirevision="patch-camera",
        legend=dict(orientation="h", x=0.35, y=1.08),
        sliders=[dict(
            active=0, x=0.13, len=0.87, y=-0.05,
            currentvalue=dict(prefix="Training step: "),
            steps=[dict(
                method="animate", label=str(s.step), args=[[str(s.step)], dict(
                    mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                )],
            ) for s in history],
        )],
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.05, showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(
                    fromcurrent=True, mode="immediate", frame=dict(duration=180, redraw=True),
                    transition=dict(duration=0),
                )]),
                dict(label="Pause", method="animate", args=[[None], dict(
                    mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                )]),
            ],
        )],
    )
    return figure


# %% Run training
if __name__ == "__main__":
    settings = Settings()
    experiment = train_experiment(settings)
    model = experiment.model
    decoders = model.decoders
    encoders = model.encoders
    history = experiment.history
    print(f"Device: {experiment.device}; {N_PATCHES} independent 2D -> 3D decoders")
    print(f"Training points: {settings.points_per_patch} per patch, {N_PATCHES * settings.points_per_patch} combined")
    print(f"Fixed-probe Sinkhorn: {history[0].loss:.5f} -> {history[-1].loss:.5f}")
    print(f"Learned shared singular scale S: {model.singular_scale.detach().item():.5f}")
    print(f"Fixed-probe encoder MSE: {history[0].encoder_loss:.5f} -> {history[-1].encoder_loss:.5f}")
    print(f"Fixed-probe reconstruction MSE: {history[0].reconstruction_loss:.5f} -> {history[-1].reconstruction_loss:.5f}")


# %% Save and open the replay
if __name__ == "__main__":
    fig = make_history_plot(experiment)
    if "__file__" in globals():
        root = Path(__file__).resolve().parents[2]
    else:
        root = next((p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "pyproject.toml").exists()), Path.cwd())
    html_path = root / "reports" / "sinkhorn_patches_with_encoder.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(
    html_path,
    include_plotlyjs=True,
    full_html=True,
    auto_play=False,
    auto_open=True,
)
print(f"Saved interactive history: {html_path.resolve()}")

# %%
