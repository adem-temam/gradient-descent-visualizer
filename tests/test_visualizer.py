"""Unit tests for Plotly visualization generation."""

from gd_visualizer.functions import get_surface
from gd_visualizer.optimizers import SGD, Adam
from gd_visualizer.runner import run_benchmark
from gd_visualizer.visualizer import (
    plot_2d_contour,
    plot_3d_surface,
    plot_distance_curves,
    plot_loss_curves,
)


def test_visualizer_figures():
    """Verify that all visualization generators construct valid Plotly figures."""
    surface = get_surface("rosenbrock")
    opts = [SGD(lr=0.001), Adam(lr=0.01)]
    results = run_benchmark(surface, opts, start_pos=(-1.0, 1.0), max_steps=10)

    fig_2d = plot_2d_contour(surface, results, grid_res=30, show_vector_field=True)
    assert fig_2d is not None
    assert len(fig_2d.data) > 0

    fig_3d = plot_3d_surface(surface, results, grid_res=20)
    assert fig_3d is not None
    assert len(fig_3d.data) > 0

    fig_loss = plot_loss_curves(results)
    assert fig_loss is not None
    assert len(fig_loss.data) == 2

    fig_dist = plot_distance_curves(results)
    assert fig_dist is not None
    assert len(fig_dist.data) == 2
