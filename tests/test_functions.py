"""Unit tests for benchmark loss surfaces and analytical gradients.

Validates that analytical gradients match central finite difference approximations
across multiple test coordinates for all registered loss surfaces.
"""

import numpy as np
import pytest
from gd_visualizer.functions import SURFACES, get_surface


def numerical_gradient(func, x: float, y: float, eps: float = 1e-6) -> np.ndarray:
    """Compute 2D numerical gradient using central finite differences."""
    df_dx = (func(x + eps, y) - func(x - eps, y)) / (2.0 * eps)
    df_dy = (func(x, y + eps) - func(x, y - eps)) / (2.0 * eps)
    return np.array([df_dx, df_dy], dtype=np.float64)


@pytest.mark.parametrize("surface_id", list(SURFACES.keys()))
def test_surface_gradient_accuracy(surface_id: str):
    """Verify analytical gradient matches central finite differences."""
    surface = get_surface(surface_id)

    # Test at 5 representative sample points within surface domain
    x_min, x_max = surface.domain_x
    y_min, y_max = surface.domain_y

    test_points = [
        (surface.default_start[0], surface.default_start[1]),
        (0.2 * x_min + 0.8 * x_max, 0.4 * y_min + 0.6 * y_max),
        (0.7 * x_min + 0.3 * x_max, 0.1 * y_min + 0.9 * y_max),
        (0.5 * (x_min + x_max), 0.5 * (y_min + y_max)),
        (surface.global_minima[0][0] + 0.3, surface.global_minima[0][1] - 0.3),
    ]

    for x, y in test_points:
        grad_analytical = surface.gradient(x, y)
        grad_numerical = numerical_gradient(surface.compute, x, y)

        diff = np.linalg.norm(grad_analytical - grad_numerical)
        norm_ref = max(np.linalg.norm(grad_analytical), 1.0)
        rel_error = diff / norm_ref

        assert rel_error < 1e-4, (
            f"Gradient mismatch for {surface_id} at ({x:.3f}, {y:.3f}): "
            f"analytical={grad_analytical}, numerical={grad_numerical}, rel_err={rel_error:.2e}"
        )


@pytest.mark.parametrize("surface_id", list(SURFACES.keys()))
def test_global_minimum_value(surface_id: str):
    """Verify that evaluating the surface at known global minima yields minimal loss."""
    surface = get_surface(surface_id)
    for min_x, min_y in surface.global_minima:
        val = surface.compute(min_x, min_y)
        grad = surface.gradient(min_x, min_y)
        assert np.isclose(val, surface.global_min_val, atol=1e-4), (
            f"{surface_id} at minimum ({min_x}, {min_y}): computed={val}, expected={surface.global_min_val}"
        )
        assert np.linalg.norm(grad) < 1e-3, (
            f"{surface_id} gradient at minimum ({min_x}, {min_y}) is non-zero: {grad}"
        )


def test_invalid_surface_name():
    """Verify KeyError is raised when requesting an unknown surface."""
    with pytest.raises(KeyError):
        get_surface("non_existent_surface")
