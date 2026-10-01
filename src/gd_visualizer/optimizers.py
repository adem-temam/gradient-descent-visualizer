"""First-order gradient-based optimization algorithms.

Implements standard optimizers used in machine learning and numerical optimization
with explicit state tracking for trajectory visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for all first-order optimizers."""

    def __init__(self, lr: float = 0.01, name: str = "Optimizer", color: str = "#3b82f6") -> None:
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = float(lr)
        self.name = name
        self.color = color
        self.step_count = 0

    @abstractmethod
    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        """Perform one optimization step.

        Args:
            position: Current parameter vector [x, y], shape (2,).
            grad_fn: Callable taking (x, y) and returning gradient vector [df/dx, df/dy].

        Returns:
            Updated parameter vector [x_new, y_new], shape (2,).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator states (e.g. velocity, moments)."""
        self.step_count = 0


class SGD(BaseOptimizer):
    """Standard vanilla gradient descent: θ_{t+1} = θ_t - lr * ∇f(θ_t)."""

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr=lr, name="SGD", color="#ef4444")

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        grad = grad_fn(pos[0], pos[1])
        return pos - self.lr * grad

    def reset(self) -> None:
        super().reset()


class Momentum(BaseOptimizer):
    """Classical Polyak momentum:
    v_{t+1} = β * v_t + lr * ∇f(θ_t)
    θ_{t+1} = θ_t - v_{t+1}
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr=lr, name="Momentum", color="#f97316")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Momentum beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.velocity: np.ndarray | None = None

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        if self.velocity is None:
            self.velocity = np.zeros_like(pos)

        grad = grad_fn(pos[0], pos[1])
        self.velocity = self.beta * self.velocity + self.lr * grad
        return pos - self.velocity

    def reset(self) -> None:
        super().reset()
        self.velocity = None


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient (lookahead momentum):
    θ_{lookahead} = θ_t - β * v_t
    v_{t+1} = β * v_t + lr * ∇f(θ_{lookahead})
    θ_{t+1} = θ_t - v_{t+1}
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr=lr, name="NAG (Nesterov)", color="#eab308")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Nesterov beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.velocity: np.ndarray | None = None

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        if self.velocity is None:
            self.velocity = np.zeros_like(pos)

        # Evaluate gradient at lookahead position
        lookahead = pos - self.beta * self.velocity
        grad = grad_fn(lookahead[0], lookahead[1])

        self.velocity = self.beta * self.velocity + self.lr * grad
        return pos - self.velocity

    def reset(self) -> None:
        super().reset()
        self.velocity = None


class AdaGrad(BaseOptimizer):
    """AdaGrad with cumulative squared gradients:
    G_{t+1} = G_t + g_t^2
    θ_{t+1} = θ_t - (lr / (sqrt(G_{t+1}) + ε)) * g_t
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__(lr=lr, name="AdaGrad", color="#10b981")
        self.eps = float(eps)
        self.g_squared: np.ndarray | None = None

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        if self.g_squared is None:
            self.g_squared = np.zeros_like(pos)

        grad = grad_fn(pos[0], pos[1])
        self.g_squared += grad**2
        adjusted_lr = self.lr / (np.sqrt(self.g_squared) + self.eps)
        return pos - adjusted_lr * grad

    def reset(self) -> None:
        super().reset()
        self.g_squared = None


class RMSprop(BaseOptimizer):
    """RMSprop with exponential moving average of squared gradients:
    v_{t+1} = decay * v_t + (1 - decay) * g_t^2
    θ_{t+1} = θ_t - (lr / (sqrt(v_{t+1}) + ε)) * g_t
    """

    def __init__(self, lr: float = 0.01, decay: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__(lr=lr, name="RMSprop", color="#8b5cf6")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"RMSprop decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.eps = float(eps)
        self.v: np.ndarray | None = None

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(pos)

        grad = grad_fn(pos[0], pos[1])
        self.v = self.decay * self.v + (1.0 - self.decay) * (grad**2)
        adjusted_lr = self.lr / (np.sqrt(self.v) + self.eps)
        return pos - adjusted_lr * grad

    def reset(self) -> None:
        super().reset()
        self.v = None


class Adam(BaseOptimizer):
    """Adam (Adaptive Moment Estimation) with bias correction:
    m_{t+1} = β1 * m_t + (1 - β1) * g_t
    v_{t+1} = β2 * v_t + (1 - β2) * g_t^2
    m̂ = m_{t+1} / (1 - β1^{t+1})
    v̂ = v_{t+1} / (1 - β2^{t+1})
    θ_{t+1} = θ_t - (lr / (sqrt(v̂) + ε)) * m̂
    """

    def __init__(
        self,
        lr: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr, name="Adam", color="#06b6d4")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Adam beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Adam beta2 must be in [0, 1), got {beta2}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def step(self, position: np.ndarray, grad_fn: Any) -> np.ndarray:
        self.step_count += 1
        pos = np.asarray(position, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(pos)
            self.v = np.zeros_like(pos)

        grad = grad_fn(pos[0], pos[1])
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1**self.step_count)
        v_hat = self.v / (1.0 - self.beta2**self.step_count)

        step_update = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return pos - step_update

    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None


OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "adagrad": AdaGrad,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def create_optimizer(optimizer_id: str, **kwargs) -> BaseOptimizer:
    """Instantiate an optimizer by its string key with optional hyperparameter overrides."""
    key = optimizer_id.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise KeyError(
            f"Optimizer '{optimizer_id}' not found. Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[key](**kwargs)
