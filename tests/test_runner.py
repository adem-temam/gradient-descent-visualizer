"""Unit tests for trajectory simulation engine and metrics."""

import numpy as np
from gd_visualizer.functions import get_surface
from gd_visualizer.optimizers import SGD, Adam
from gd_visualizer.runner import run_benchmark, run_optimizer


def test_run_optimizer_basic():
    """Verify single optimizer run records steps properly."""
    surface = get_surface("sphere")
    opt = SGD(lr=0.05)
    result = run_optimizer(surface, opt, start_pos=(1.0, 1.0), max_steps=10)

    assert result.optimizer_name == "SGD"
    assert len(result.points) == 11  # step 0 to 10
    assert result.points[0].step == 0
    assert result.points[-1].step == 10
    assert result.final_loss < result.points[0].loss


def test_run_optimizer_divergence_handling():
    """Verify runaway learning rates trigger divergence flag without crashing."""
    surface = get_surface("rosenbrock")
    opt = SGD(lr=10.0)  # Intentionally massive learning rate to cause divergence
    result = run_optimizer(surface, opt, start_pos=(1.5, 1.5), max_steps=50)

    assert result.diverged is True
    assert "Diverged" in result.termination_reason or "overflow" in result.termination_reason


def test_run_benchmark_multi():
    """Verify multi-optimizer benchmark runs all instances."""
    surface = get_surface("quadratic_bowl")
    opts = [SGD(lr=0.01), Adam(lr=0.05)]
    results = run_benchmark(surface, opts, start_pos=(-2.0, 2.0), max_steps=20)

    assert "SGD" in results
    assert "Adam" in results
    assert results["SGD"].steps_taken == 20
    assert results["Adam"].steps_taken == 20
