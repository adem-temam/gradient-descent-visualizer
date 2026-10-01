"""Plotly-based visualization components for 2D contours, 3D loss surfaces, and convergence metrics."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gd_visualizer.functions import LossSurface
from gd_visualizer.runner import TrajectoryResult


def plot_2d_contour(
    surface: LossSurface,
    results: dict[str, TrajectoryResult] | None = None,
    grid_res: int = 150,
    show_vector_field: bool = False,
    log_scale_contours: bool = True,
) -> go.Figure:
    """Generate an interactive 2D contour plot with optimizer trajectory overlays."""
    x_min, x_max = surface.domain_x
    y_min, y_max = surface.domain_y

    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = surface.compute(X, Y)

    # Use log-scaled values for contour levels if dynamic range is large
    Z_plot = np.log1p(np.maximum(Z - surface.global_min_val, 0.0)) if log_scale_contours else Z

    fig = go.Figure()

    # 1. Base contour layer
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z_plot,
            colorscale="Viridis",
            ncontours=35,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.7, color="rgba(255, 255, 255, 0.3)"),
            colorbar=dict(
                title=dict(
                    text="log(1 + f - f*)" if log_scale_contours else "f(x, y)",
                    side="top",
                    font=dict(size=11, color="#cbd5e1"),
                ),
                x=1.02,
                len=0.9,
                tickfont=dict(color="#94a3b8", size=10),
            ),
            hoverinfo="x+y+z",
            showscale=True,
        )
    )

    # 2. Vector field (optional gradient direction arrows)
    if show_vector_field:
        arrow_res = 18
        xv = np.linspace(x_min, x_max, arrow_res)
        yv = np.linspace(y_min, y_max, arrow_res)
        Xv, Yv = np.meshgrid(xv, yv)
        U = np.zeros_like(Xv)
        V = np.zeros_like(Yv)

        for i in range(arrow_res):
            for j in range(arrow_res):
                g = surface.gradient(Xv[i, j], Yv[i, j])
                norm = np.linalg.norm(g)
                if norm > 1e-6:
                    # Negative gradient normalized for visualization
                    U[i, j] = -g[0] / norm * ((x_max - x_min) / (arrow_res * 2.5))
                    V[i, j] = -g[1] / norm * ((y_max - y_min) / (arrow_res * 2.5))

        # Flatten for line segment trace
        arrow_x = []
        arrow_y = []
        for i in range(arrow_res):
            for j in range(arrow_res):
                arrow_x.extend([Xv[i, j], Xv[i, j] + U[i, j], None])
                arrow_y.extend([Yv[i, j], Yv[i, j] + V[i, j], None])

        fig.add_trace(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.25)", width=1.0),
                name="Negative Gradient Field",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 3. Known Global Minima
    min_x = [m[0] for m in surface.global_minima]
    min_y = [m[1] for m in surface.global_minima]
    fig.add_trace(
        go.Scatter(
            x=min_x,
            y=min_y,
            mode="markers",
            marker=dict(
                symbol="star",
                size=11,
                color="#facc15",
                line=dict(color="#000000", width=1.5),
            ),
            name="Global Minimum (θ*)",
            hovertemplate="<b>Global Minimum (θ*)</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
        )
    )

    # 4. Optimizer Trajectories
    if results:
        for opt_name, res in results.items():
            if not res.points:
                continue

            xs = res.x_coords
            ys = res.y_coords

            # Trajectory Path
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color=res.color, width=2.5),
                    marker=dict(size=4, color=res.color),
                    name=opt_name,
                    hovertemplate=(
                        f"<b>{opt_name}</b><br>"
                        + "Step: %{customdata[0]}<br>"
                        + "x: %{x:.4f}<br>"
                        + "y: %{y:.4f}<br>"
                        + "Loss: %{customdata[1]:.4e}<br>"
                        + "||∇f||: %{customdata[2]:.4e}"
                        + "<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        (
                            [p.step for p in res.points],
                            [p.loss for p in res.points],
                            [p.grad_norm for p in res.points],
                        )
                    ),
                )
            )

            # Start point marker
            fig.add_trace(
                go.Scatter(
                    x=[xs[0]],
                    y=[ys[0]],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=res.color,
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # End point marker
            end_symbol = "x" if res.diverged else ("circle-cross" if res.converged else "diamond")
            fig.add_trace(
                go.Scatter(
                    x=[xs[-1]],
                    y=[ys[-1]],
                    mode="markers",
                    marker=dict(
                        symbol=end_symbol,
                        size=11,
                        color=res.color,
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>2D Loss Contour & Trajectories:</b> {surface.name}",
            font=dict(size=15, color="#f8fafc"),
            x=0.01,
            y=0.98,
        ),
        xaxis=dict(
            title="x",
            range=[x_min, x_max],
            zeroline=False,
            gridcolor="#334155",
            color="#94a3b8",
        ),
        yaxis=dict(
            title="y",
            range=[y_min, y_max],
            zeroline=False,
            gridcolor="#334155",
            color="#94a3b8",
        ),
        dragmode="zoom",
        uirevision=surface.id,
        hovermode="closest",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(color="#e2e8f0", size=11),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        margin=dict(l=45, r=70, t=75, b=45),
    )
    return fig


def plot_3d_surface(
    surface: LossSurface,
    results: dict[str, TrajectoryResult] | None = None,
    grid_res: int = 70,
    z_clip_percentile: float = 95.0,
) -> go.Figure:
    """Generate an interactive 3D surface plot with 3D optimizer trajectories."""
    x_min, x_max = surface.domain_x
    y_min, y_max = surface.domain_y

    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = surface.compute(X, Y)

    # Clip extreme heights to prevent distortion of valley features
    upper_bound = float(np.percentile(Z, z_clip_percentile))
    Z_clipped = np.clip(Z, surface.global_min_val, upper_bound)

    fig = go.Figure()

    # 1. 3D Surface
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=Z_clipped,
            colorscale="Viridis",
            opacity=0.88,
            showscale=False,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#ffffff", project_z=True)
            ),
        )
    )

    # 2. 3D Trajectories
    if results:
        for opt_name, res in results.items():
            if not res.points:
                continue

            xs = res.x_coords
            ys = res.y_coords
            # Ensure trajectory points sit slightly above the surface for visibility
            zs = [float(surface.compute(x, y)) + 0.05 * (upper_bound - surface.global_min_val) for x, y in zip(xs, ys)]
            zs = [min(z, upper_bound * 1.05) for z in zs]

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines+markers",
                    line=dict(color=res.color, width=4),
                    marker=dict(size=3, color=res.color),
                    name=opt_name,
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>3D Loss Landscape:</b> {surface.name}",
            font=dict(size=16, color="#f8fafc"),
        ),
        scene=dict(
            xaxis=dict(title="x", gridcolor="#334155", color="#94a3b8", backgroundcolor="#0f172a"),
            yaxis=dict(title="y", gridcolor="#334155", color="#94a3b8", backgroundcolor="#0f172a"),
            zaxis=dict(title="f(x, y)", gridcolor="#334155", color="#94a3b8", backgroundcolor="#0f172a"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            aspectmode="cube",
            dragmode="orbit",
        ),
        uirevision=surface.id,
        paper_bgcolor="#0f172a",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(color="#e2e8f0", size=11),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=65, b=10),
    )
    return fig


def plot_loss_curves(results: dict[str, TrajectoryResult]) -> go.Figure:
    """Generate convergence plot comparing loss values over optimization steps."""
    fig = go.Figure()

    for opt_name, res in results.items():
        if not res.points:
            continue
        steps = [p.step for p in res.points]
        losses = [max(p.loss, 1e-12) for p in res.points]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode="lines",
                line=dict(color=res.color, width=2.5),
                name=opt_name,
                hovertemplate=f"<b>{opt_name}</b>: %{{y:.4e}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>Loss Convergence vs. Step</b>", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="Step", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(
            title="Objective Value f(θ)",
            type="log",
            gridcolor="#334155",
            color="#94a3b8",
        ),
        hovermode="x unified",
        uirevision="constant",
        dragmode="zoom",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        legend=dict(
            font=dict(color="#e2e8f0"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        margin=dict(l=40, r=20, t=45, b=40),
    )
    return fig


def plot_distance_curves(results: dict[str, TrajectoryResult]) -> go.Figure:
    """Generate plot tracking Euclidean distance to global minimum over time."""
    fig = go.Figure()

    for opt_name, res in results.items():
        if not res.points:
            continue
        steps = [p.step for p in res.points]
        distances = [max(p.distance_to_min, 1e-12) for p in res.points]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=distances,
                mode="lines",
                line=dict(color=res.color, width=2.5),
                name=opt_name,
                hovertemplate=f"<b>{opt_name}</b>: %{{y:.4e}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Distance to Global Minimum ||θ - θ*||</b>",
            font=dict(size=15, color="#f8fafc"),
        ),
        xaxis=dict(title="Step", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(
            title="Euclidean Distance",
            type="log",
            gridcolor="#334155",
            color="#94a3b8",
        ),
        hovermode="x unified",
        uirevision="constant",
        dragmode="zoom",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        legend=dict(
            font=dict(color="#e2e8f0"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        margin=dict(l=40, r=20, t=45, b=40),
    )
    return fig
