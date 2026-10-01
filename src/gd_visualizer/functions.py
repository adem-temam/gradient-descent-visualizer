"""Benchmark loss surfaces and analytical gradients for optimization testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class LossSurface:
    """Represents a 2D mathematical objective function for optimization benchmarking."""

    id: str
    name: str
    description: str
    formula_latex: str
    domain_x: tuple[float, float]
    domain_y: tuple[float, float]
    global_minima: list[tuple[float, float]]
    global_min_val: float
    default_start: tuple[float, float]
    func: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float]
    grad_func: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]

    def compute(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Compute the scalar objective function value at (x, y). Vectorized over numpy arrays."""
        return self.func(x, y)

    def gradient(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Compute analytical gradient [df/dx, df/dy] at (x, y)."""
        grad = self.grad_func(x, y)
        return np.asarray(grad, dtype=np.float64)


# --- Surface Definitions ---

def _rosenbrock(x, y, a: float = 1.0, b: float = 100.0):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def _rosenbrock_grad(x, y, a: float = 1.0, b: float = 100.0):
    df_dx = 2.0 * (x - a) - 4.0 * b * x * (y - x**2)
    df_dy = 2.0 * b * (y - x**2)
    return np.array([df_dx, df_dy], dtype=np.float64)


def _beale(x, y):
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * (y**2)
    t3 = 2.625 - x + x * (y**3)
    return t1**2 + t2**2 + t3**2


def _beale_grad(x, y):
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * (y**2)
    t3 = 2.625 - x + x * (y**3)
    df_dx = 2.0 * t1 * (y - 1.0) + 2.0 * t2 * (y**2 - 1.0) + 2.0 * t3 * (y**3 - 1.0)
    df_dy = 2.0 * t1 * x + 4.0 * t2 * x * y + 6.0 * t3 * x * (y**2)
    return np.array([df_dx, df_dy], dtype=np.float64)


def _rastrigin(x, y, a: float = 10.0):
    return 2.0 * a + (x**2 - a * np.cos(2.0 * np.pi * x)) + (y**2 - a * np.cos(2.0 * np.pi * y))


def _rastrigin_grad(x, y, a: float = 10.0):
    df_dx = 2.0 * x + 2.0 * np.pi * a * np.sin(2.0 * np.pi * x)
    df_dy = 2.0 * y + 2.0 * np.pi * a * np.sin(2.0 * np.pi * y)
    return np.array([df_dx, df_dy], dtype=np.float64)


def _saddle(x, y):
    # Escapable double-well saddle: saddle at (0, 0), minima at (0, ±sqrt(2))
    return x**2 - y**2 + 0.25 * (y**4)


def _saddle_grad(x, y):
    df_dx = 2.0 * x
    df_dy = -2.0 * y + (y**3)
    return np.array([df_dx, df_dy], dtype=np.float64)


def _himmelblau(x, y):
    return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2


def _himmelblau_grad(x, y):
    t1 = x**2 + y - 11.0
    t2 = x + y**2 - 7.0
    df_dx = 4.0 * x * t1 + 2.0 * t2
    df_dy = 2.0 * t1 + 4.0 * y * t2
    return np.array([df_dx, df_dy], dtype=np.float64)


def _quadratic_bowl(x, y, condition_number: float = 20.0):
    # High condition number ellipse: ill-conditioned ravine
    return 0.5 * (x**2 + condition_number * (y**2))


def _quadratic_bowl_grad(x, y, condition_number: float = 20.0):
    df_dx = x
    df_dy = condition_number * y
    return np.array([df_dx, df_dy], dtype=np.float64)


def _sphere(x, y):
    return x**2 + y**2


def _sphere_grad(x, y):
    return np.array([2.0 * x, 2.0 * y], dtype=np.float64)


SURFACES: dict[str, LossSurface] = {
    "rosenbrock": LossSurface(
        id="rosenbrock",
        name="Rosenbrock (Banana)",
        description="A notoriously ill-conditioned curved valley where the gradient is almost orthogonal to the valley path. Tests momentum and adaptive learning rates.",
        formula_latex=r"f(x, y) = (1 - x)^2 + 100(y - x^2)^2",
        domain_x=(-2.0, 2.0),
        domain_y=(-1.0, 3.0),
        global_minima=[(1.0, 1.0)],
        global_min_val=0.0,
        default_start=(-1.5, 2.0),
        func=_rosenbrock,
        grad_func=_rosenbrock_grad,
    ),
    "beale": LossSurface(
        id="beale",
        name="Beale's Function",
        description="A surface with sharp steep ridges at the periphery and a flat plateau near the minimum. Tests how optimizers handle vanishing gradients.",
        formula_latex=r"f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        domain_x=(-4.5, 4.5),
        domain_y=(-4.5, 4.5),
        global_minima=[(3.0, 0.5)],
        global_min_val=0.0,
        default_start=(-2.0, 2.0),
        func=_beale,
        grad_func=_beale_grad,
    ),
    "saddle": LossSurface(
        id="saddle",
        name="Escapable Saddle Point",
        description="A landscape with a saddle point at the origin (0, 0) and two symmetrical global minima at (0, ±√2). Demonstrates how optimizers escape zero-gradient points.",
        formula_latex=r"f(x, y) = x^2 - y^2 + \frac{1}{4}y^4",
        domain_x=(-2.5, 2.5),
        domain_y=(-2.5, 2.5),
        global_minima=[(0.0, np.sqrt(2)), (0.0, -np.sqrt(2))],
        global_min_val=-1.0,
        default_start=(0.05, 0.1),
        func=_saddle,
        grad_func=_saddle_grad,
    ),
    "quadratic_bowl": LossSurface(
        id="quadratic_bowl",
        name="Ill-Conditioned Quadratic",
        description="An elongated quadratic valley with an anisotropic condition number (κ=20). Vanilla SGD oscillates violently between walls while momentum speeds along the valley.",
        formula_latex=r"f(x, y) = \frac{1}{2}(x^2 + 20y^2)",
        domain_x=(-5.0, 5.0),
        domain_y=(-5.0, 5.0),
        global_minima=[(0.0, 0.0)],
        global_min_val=0.0,
        default_start=(-4.0, 4.0),
        func=_quadratic_bowl,
        grad_func=_quadratic_bowl_grad,
    ),
    "himmelblau": LossSurface(
        id="himmelblau",
        name="Himmelblau's Function",
        description="A multimodal landscape with four identical global minima separated by ridges. Shows which basin of attraction an optimizer selects.",
        formula_latex=r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        domain_x=(-5.0, 5.0),
        domain_y=(-5.0, 5.0),
        global_minima=[
            (3.0, 2.0),
            (-2.805118, 3.131312),
            (-3.779310, -3.283186),
            (3.584428, -1.848126),
        ],
        global_min_val=0.0,
        default_start=(-0.5, 0.0),
        func=_himmelblau,
        grad_func=_himmelblau_grad,
    ),
    "rastrigin": LossSurface(
        id="rastrigin",
        name="Rastrigin Function",
        description="A highly non-convex surface with a regular cosine-modulated grid of local minima. Demonstrates vulnerability of gradient methods to local traps.",
        formula_latex=r"f(x, y) = 20 + x^2 - 10\cos(2\pi x) + y^2 - 10\cos(2\pi y)",
        domain_x=(-5.12, 5.12),
        domain_y=(-5.12, 5.12),
        global_minima=[(0.0, 0.0)],
        global_min_val=0.0,
        default_start=(-2.5, -2.5),
        func=_rastrigin,
        grad_func=_rastrigin_grad,
    ),
    "sphere": LossSurface(
        id="sphere",
        name="Convex Sphere",
        description="A perfectly isotropic quadratic bowl with condition number 1. Acts as a baseline calibration test for optimizer convergence.",
        formula_latex=r"f(x, y) = x^2 + y^2",
        domain_x=(-3.0, 3.0),
        domain_y=(-3.0, 3.0),
        global_minima=[(0.0, 0.0)],
        global_min_val=0.0,
        default_start=(-2.5, 2.5),
        func=_sphere,
        grad_func=_sphere_grad,
    ),
}


def get_surface(surface_id: str) -> LossSurface:
    """Retrieve a benchmark surface by ID."""
    if surface_id not in SURFACES:
        raise KeyError(
            f"Surface '{surface_id}' not found. Available: {list(SURFACES.keys())}"
        )
    return SURFACES[surface_id]
