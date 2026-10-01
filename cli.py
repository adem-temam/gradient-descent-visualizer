#!/usr/bin/env python3
"""Command-line interface for running headless optimization benchmarks and exporting plots."""

from __future__ import annotations

import argparse
import sys
from gd_visualizer.functions import SURFACES, get_surface
from gd_visualizer.optimizers import OPTIMIZER_REGISTRY, create_optimizer
from gd_visualizer.runner import run_benchmark
from gd_visualizer.visualizer import plot_2d_contour


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-order gradient descent simulations on benchmark loss surfaces."
    )
    parser.add_argument(
        "--surface",
        type=str,
        default="rosenbrock",
        choices=list(SURFACES.keys()),
        help="Benchmark loss surface to optimize on (default: rosenbrock)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=["sgd", "momentum", "rmsprop", "adam"],
        choices=list(OPTIMIZER_REGISTRY.keys()),
        help="List of optimizers to compare (default: sgd momentum rmsprop adam)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Maximum optimization steps (default: 150)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Global learning rate override for all optimizers",
    )
    parser.add_argument(
        "--start-x",
        type=float,
        default=None,
        help="Starting coordinate X (defaults to surface preset)",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=None,
        help="Starting coordinate Y (defaults to surface preset)",
    )
    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="Optional path to export interactive 2D contour HTML visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surface = get_surface(args.surface)

    # Determine starting coordinates
    start_x = args.start_x if args.start_x is not None else surface.default_start[0]
    start_y = args.start_y if args.start_y is not None else surface.default_start[1]
    start_pos = (start_x, start_y)

    print("=" * 70)
    print(f" Gradient Descent Visualizer Benchmark: {surface.name}")
    print(f" Starting point: ({start_x:.3f}, {start_y:.3f}) | Max steps: {args.steps}")
    print("=" * 70)

    # Instantiate optimizers
    optimizers = []
    for opt_id in args.optimizers:
        kwargs = {}
        if args.lr is not None:
            kwargs["lr"] = args.lr
        optimizers.append(create_optimizer(opt_id, **kwargs))

    # Run benchmark
    results = run_benchmark(
        surface=surface,
        optimizers=optimizers,
        start_pos=start_pos,
        max_steps=args.steps,
    )

    # Display comparison table
    header = f"{'Optimizer':<14} | {'Steps':<6} | {'Final Loss':<14} | {'Min Loss':<14} | {'Status'}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        status = "CONVERGED" if res.converged else ("DIVERGED" if res.diverged else "MAX_STEPS")
        print(
            f"{name:<14} | {res.steps_taken:<6} | {res.final_loss:<14.4e} | {res.min_loss:<14.4e} | {status}"
        )
    print("=" * 70)

    # Optional HTML export
    if args.export_html:
        fig = plot_2d_contour(surface, results, grid_res=150, show_vector_field=True)
        fig.write_html(args.export_html)
        print(f"✓ Interactive contour visualization exported to: {args.export_html}")


if __name__ == "__main__":
    main()
