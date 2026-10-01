"""Unit tests for first-order optimizers.

Verifies convergence properties on quadratic and convex surfaces,
state resetting, and parameter validation.
"""

import numpy as np
import pytest
from gd_visualizer.functions import get_surface
from gd_visualizer.optimizers import (
    OPTIMIZER_REGISTRY,
    SGD,
    Adam,
    Momentum,
    create_optimizer,
)


@pytest.mark.parametrize("opt_id", list(OPTIMIZER_REGISTRY.keys()))
def test_optimizer_convergence_on_sphere(opt_id: str):
    """Verify that every optimizer converges toward (0, 0) on the convex sphere function."""
    surface = get_surface("sphere")
    start_pos = np.array([2.0, 2.0], dtype=np.float64)

    # Use learning rates tuned for 150 steps convergence on sphere
    tuned_lrs = {
        "sgd": 0.05,
        "momentum": 0.03,
        "nag": 0.03,
        "adagrad": 0.5,
        "rmsprop": 0.08,
        "adam": 0.1,
    }

    opt = create_optimizer(opt_id, lr=tuned_lrs[opt_id])
    pos = start_pos.copy()
    initial_loss = surface.compute(pos[0], pos[1])

    for _ in range(150):
        pos = opt.step(pos, surface.gradient)

    final_loss = surface.compute(pos[0], pos[1])
    # Expect loss reduction of at least 95%
    assert final_loss < 0.05 * initial_loss, (
        f"{opt_id} failed to converge on sphere: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_optimizer_reset():
    """Verify reset clears internal optimizer history."""
    surface = get_surface("sphere")
    adam = Adam(lr=0.01)
    pos = np.array([1.0, 1.0])

    # Run 5 steps
    for _ in range(5):
        pos = adam.step(pos, surface.gradient)

    assert adam.step_count == 5
    assert adam.m is not None
    assert adam.v is not None

    adam.reset()
    assert adam.step_count == 0
    assert adam.m is None
    assert adam.v is None


def test_invalid_hyperparameters():
    """Verify exception raising for invalid hyperparameter inputs."""
    with pytest.raises(ValueError):
        SGD(lr=-0.01)

    with pytest.raises(ValueError):
        Momentum(lr=0.01, beta=1.5)

    with pytest.raises(ValueError):
        Adam(lr=0.01, beta1=-0.1)

    with pytest.raises(KeyError):
        create_optimizer("non_existent_optimizer")
