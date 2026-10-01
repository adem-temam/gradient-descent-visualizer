"""Trajectory simulation engine and performance metrics tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from gd_visualizer.functions import LossSurface
from gd_visualizer.optimizers import BaseOptimizer


@dataclass
class TrajectoryPoint:
    """Snapshot of optimization state at a single step."""

    step: int
    x: float
    y: float
    loss: float
    grad_norm: float
    distance_to_min: float


@dataclass
class TrajectoryResult:
    """Complete trajectory trace and performance summary for an optimizer run."""

    optimizer_name: str
    color: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    converged: bool = False
    diverged: bool = False
    termination_reason: str = ""

    @property
    def x_coords(self) -> list[float]:
        return [p.x for p in self.points]

    @property
    def y_coords(self) -> list[float]:
        return [p.y for p in self.points]

    @property
    def losses(self) -> list[float]:
        return [p.loss for p in self.points]

    @property
    def grad_norms(self) -> list[float]:
        return [p.grad_norm for p in self.points]

    @property
    def final_loss(self) -> float:
        return self.points[-1].loss if self.points else float("nan")

    @property
    def min_loss(self) -> float:
        return min((p.loss for p in self.points), default=float("nan"))

    @property
    def steps_taken(self) -> int:
        return len(self.points) - 1 if self.points else 0


def _distance_to_nearest_minimum(
    x: float, y: float, minima: list[tuple[float, float]]
) -> float:
    return min(np.hypot(x - mx, y - my) for mx, my in minima)


def run_optimizer(
    surface: LossSurface,
    optimizer: BaseOptimizer,
    start_pos: tuple[float, float],
    max_steps: int = 150,
    tol: float = 1e-6,
    divergence_limit: float = 1e5,
) -> TrajectoryResult:
    """Execute optimization on a surface with safety bounds and convergence detection.

    Args:
        surface: The objective loss surface.
        optimizer: Configured optimizer instance.
        start_pos: Starting coordinates (x0, y0).
        max_steps: Maximum iterations before stopping.
        tol: Gradient norm tolerance for declaring convergence.
        divergence_limit: Maximum allowed coordinate magnitude before declaring divergence.

    Returns:
        TrajectoryResult containing step-by-step points and status flags.
    """
    optimizer.reset()
    pos = np.array(start_pos, dtype=np.float64)

    initial_loss = float(surface.compute(pos[0], pos[1]))
    initial_grad = surface.gradient(pos[0], pos[1])
    initial_norm = float(np.linalg.norm(initial_grad))
    initial_dist = _distance_to_nearest_minimum(pos[0], pos[1], surface.global_minima)

    result = TrajectoryResult(optimizer_name=optimizer.name, color=optimizer.color)
    result.points.append(
        TrajectoryPoint(
            step=0,
            x=float(pos[0]),
            y=float(pos[1]),
            loss=initial_loss,
            grad_norm=initial_norm,
            distance_to_min=initial_dist,
        )
    )

    for step_num in range(1, max_steps + 1):
        # Check convergence
        if initial_norm < tol:
            result.converged = True
            result.termination_reason = f"Converged: ||∇f|| < {tol:.1e}"
            break

        # Optimizer step
        new_pos = optimizer.step(pos, surface.gradient)

        # Check numerical validity
        if np.any(np.isnan(new_pos)) or np.any(np.isinf(new_pos)):
            result.diverged = True
            result.termination_reason = "Numerical overflow (NaN/Inf detected)"
            break

        if np.max(np.abs(new_pos)) > divergence_limit:
            result.diverged = True
            result.termination_reason = f"Diverged: coordinates exceeded {divergence_limit:.0e}"
            break

        pos = new_pos
        loss = float(surface.compute(pos[0], pos[1]))
        grad = surface.gradient(pos[0], pos[1])
        grad_norm = float(np.linalg.norm(grad))
        dist = _distance_to_nearest_minimum(pos[0], pos[1], surface.global_minima)

        result.points.append(
            TrajectoryPoint(
                step=step_num,
                x=float(pos[0]),
                y=float(pos[1]),
                loss=loss,
                grad_norm=grad_norm,
                distance_to_min=dist,
            )
        )

        if grad_norm < tol:
            result.converged = True
            result.termination_reason = f"Converged at step {step_num} (||∇f|| < {tol:.1e})"
            break

    if not result.converged and not result.diverged:
        result.termination_reason = f"Reached maximum steps ({max_steps})"

    return result


def run_benchmark(
    surface: LossSurface,
    optimizers: list[BaseOptimizer],
    start_pos: tuple[float, float],
    max_steps: int = 150,
) -> dict[str, TrajectoryResult]:
    """Run multiple optimizers on the same surface from the same initial coordinates."""
    results: dict[str, TrajectoryResult] = {}
    for opt in optimizers:
        results[opt.name] = run_optimizer(
            surface=surface,
            optimizer=opt,
            start_pos=start_pos,
            max_steps=max_steps,
        )
    return results
