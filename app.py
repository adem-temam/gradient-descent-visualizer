"""Gradient Descent Visualizer - Interactive Streamlit Application.

An interactive dashboard for comparing first-order optimization algorithms
across classic non-convex and ill-conditioned loss surfaces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gd_visualizer.functions import SURFACES, get_surface
from gd_visualizer.optimizers import OPTIMIZER_REGISTRY, create_optimizer
from gd_visualizer.runner import run_benchmark
from gd_visualizer.visualizer import (
    plot_2d_contour,
    plot_3d_surface,
    plot_distance_curves,
    plot_loss_curves,
)

st.set_page_config(
    page_title="Gradient Descent Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom plot interaction config: enables smooth scroll zooming, disables selection tools
PLOT_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

# Custom styling for clean, professional layout
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 6px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
    }
    .guide-section {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "banana_ravine": {
        "name": "Rosenbrock Valley",
        "surface": "rosenbrock",
        "opts": ["sgd", "momentum", "rmsprop", "adam"],
        "lr": 0.005,
        "steps": 120,
        "x": -1.5,
        "y": 1.5,
        "desc": "SGD oscillates across steep ravine walls, while Adam and Momentum accelerate along the parabolic valley.",
    },
    "saddle_escape": {
        "name": "Saddle Point Escape",
        "surface": "saddle",
        "opts": ["sgd", "momentum", "nag", "adam"],
        "lr": 0.03,
        "steps": 90,
        "x": 0.02,
        "y": 0.05,
        "desc": "Vanilla SGD stalls at the zero-gradient saddle (0, 0), whereas Momentum and Adam build velocity to escape along negative curvature.",
    },
    "anisotropy_bowl": {
        "name": "Ill-Conditioned Quadratic",
        "surface": "quadratic_bowl",
        "opts": ["sgd", "momentum", "adam"],
        "lr": 0.04,
        "steps": 70,
        "x": -4.0,
        "y": 4.0,
        "desc": "High condition number (kappa=20). Gradient in y is 20x steeper than x. Momentum dampens orthogonal bounces and speeds to the origin.",
    },
    "multimodal_maze": {
        "name": "Rastrigin Multimodal",
        "surface": "rastrigin",
        "opts": ["sgd", "momentum", "adam"],
        "lr": 0.02,
        "steps": 100,
        "x": -2.0,
        "y": -2.0,
        "desc": "High frequency cosine modulation creates dense local minima traps, illustrating vulnerability of first-order methods to non-convex basins.",
    },
}


def main() -> None:
    st.title("Gradient Descent Visualizer")
    st.caption(
        "Interactive numerical optimization and trajectory benchmarking across non-convex and ill-conditioned loss surfaces."
    )

    # Initialize preset session state defaults
    if "surface_id" not in st.session_state:
        st.session_state["surface_id"] = "rosenbrock"
    if "selected_opts" not in st.session_state:
        st.session_state["selected_opts"] = ["sgd", "momentum", "rmsprop", "adam"]
    if "max_steps" not in st.session_state:
        st.session_state["max_steps"] = 120
    if "custom_lr" not in st.session_state:
        st.session_state["custom_lr"] = 0.01
    if "start_x" not in st.session_state:
        st.session_state["start_x"] = -1.5
    if "start_y" not in st.session_state:
        st.session_state["start_y"] = 2.0

    # --- Sidebar Configuration ---
    st.sidebar.header("Benchmark Presets")
    st.sidebar.caption("Load canonical optimization benchmarks:")

    col_p1, col_p2 = st.sidebar.columns(2)
    with col_p1:
        if st.button("Rosenbrock Valley", use_container_width=True):
            p = PRESETS["banana_ravine"]
            st.session_state["surface_id"] = p["surface"]
            st.session_state["selected_opts"] = p["opts"]
            st.session_state["custom_lr"] = p["lr"]
            st.session_state["max_steps"] = p["steps"]
            st.session_state["start_x"] = p["x"]
            st.session_state["start_y"] = p["y"]
            st.rerun()

        if col_p1.button("High Anisotropy", use_container_width=True):
            p = PRESETS["anisotropy_bowl"]
            st.session_state["surface_id"] = p["surface"]
            st.session_state["selected_opts"] = p["opts"]
            st.session_state["custom_lr"] = p["lr"]
            st.session_state["max_steps"] = p["steps"]
            st.session_state["start_x"] = p["x"]
            st.session_state["start_y"] = p["y"]
            st.rerun()

    with col_p2:
        if st.button("Saddle Escape", use_container_width=True):
            p = PRESETS["saddle_escape"]
            st.session_state["surface_id"] = p["surface"]
            st.session_state["selected_opts"] = p["opts"]
            st.session_state["custom_lr"] = p["lr"]
            st.session_state["max_steps"] = p["steps"]
            st.session_state["start_x"] = p["x"]
            st.session_state["start_y"] = p["y"]
            st.rerun()

        if col_p2.button("Rastrigin Maze", use_container_width=True):
            p = PRESETS["multimodal_maze"]
            st.session_state["surface_id"] = p["surface"]
            st.session_state["selected_opts"] = p["opts"]
            st.session_state["custom_lr"] = p["lr"]
            st.session_state["max_steps"] = p["steps"]
            st.session_state["start_x"] = p["x"]
            st.session_state["start_y"] = p["y"]
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Loss Landscape")
    surface_options = {s_id: s.name for s_id, s in SURFACES.items()}
    current_surf_index = (
        list(surface_options.keys()).index(st.session_state["surface_id"])
        if st.session_state["surface_id"] in surface_options
        else 0
    )

    selected_surface_id = st.sidebar.selectbox(
        "Landscape Selection",
        options=list(surface_options.keys()),
        format_func=lambda x: surface_options[x],
        index=current_surf_index,
    )
    st.session_state["surface_id"] = selected_surface_id
    surface = get_surface(selected_surface_id)

    st.sidebar.markdown("**Mathematical Formulation:**")
    st.sidebar.latex(surface.formula_latex)
    st.sidebar.info(surface.description)

    st.sidebar.header("Optimization Configuration")

    all_opt_keys = list(OPTIMIZER_REGISTRY.keys())
    valid_selected_opts = [
        o for o in st.session_state["selected_opts"] if o in all_opt_keys
    ]
    if not valid_selected_opts:
        valid_selected_opts = ["sgd", "momentum", "rmsprop", "adam"]

    selected_opts = st.sidebar.multiselect(
        "Active Optimizers",
        options=all_opt_keys,
        default=valid_selected_opts,
        format_func=lambda x: OPTIMIZER_REGISTRY[x]().name,
    )
    st.session_state["selected_opts"] = selected_opts

    if not selected_opts:
        st.warning("Please select at least one optimizer from the sidebar.")
        return

    col_steps, col_lr = st.sidebar.columns(2)
    with col_steps:
        max_steps = st.slider(
            "Max Steps",
            min_value=10,
            max_value=400,
            value=int(st.session_state["max_steps"]),
            step=10,
        )
        st.session_state["max_steps"] = max_steps

    with col_lr:
        custom_lr = st.number_input(
            "Learning Rate (alpha)",
            min_value=0.0001,
            max_value=2.0,
            value=float(st.session_state["custom_lr"]),
            step=0.005,
            format="%.4f",
        )
        st.session_state["custom_lr"] = custom_lr

    st.sidebar.subheader("Initial Coordinates (x0, y0)")
    col_x, col_y = st.sidebar.columns(2)
    x_min, x_max = surface.domain_x
    y_min, y_max = surface.domain_y

    init_x = float(
        np.clip(
            st.session_state.get("start_x", surface.default_start[0]),
            x_min,
            x_max,
        )
    )
    init_y = float(
        np.clip(
            st.session_state.get("start_y", surface.default_start[1]),
            y_min,
            y_max,
        )
    )

    with col_x:
        start_x = st.slider(
            "x0",
            min_value=float(x_min),
            max_value=float(x_max),
            value=init_x,
            step=0.05,
        )
        st.session_state["start_x"] = start_x

    with col_y:
        start_y = st.slider(
            "y0",
            min_value=float(y_min),
            max_value=float(y_max),
            value=init_y,
            step=0.05,
        )
        st.session_state["start_y"] = start_y

    st.sidebar.subheader("Display Options")
    show_vector_field = st.sidebar.checkbox("Show Gradient Field (Quivers)", value=False)
    log_scale_contours = st.sidebar.checkbox("Logarithmic Contours", value=True)

    # Run Simulation
    optimizers = [create_optimizer(opt_id, lr=custom_lr) for opt_id in selected_opts]
    results = run_benchmark(
        surface=surface,
        optimizers=optimizers,
        start_pos=(start_x, start_y),
        max_steps=max_steps,
    )

    # --- Main Content Tabs ---
    tab_guide, tab_2d, tab_3d, tab_convergence, tab_metrics, tab_mechanics = st.tabs(
        [
            "Overview & User Guide",
            "2D Contour Map",
            "3D Surface Landscape",
            "Convergence Analysis",
            "Performance Metrics",
            "Mathematical Formulations",
        ]
    )

    with tab_guide:
        st.subheader("System Overview and User Guide")
        st.markdown(
            """
            This platform simulates and benchmarks first-order gradient optimization methods 
            on mathematically defined objective manifolds. It provides side-by-side trajectory 
            comparisons, rate-of-convergence diagnostics, and numerical stability tracking.
            """
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(
                """
                ### Controls and Parameter Definitions
                - **Benchmark Presets**: One-click configurations loading canonical landscape experiments.
                - **Loss Landscape**: Selects objective function $f(x, y)$. Surfaces test specific geometric hurdles:
                  anisotropic ravines, flat plateaus, zero-gradient saddle points, and multi-modal basins.
                - **Active Optimizers**: Selects candidate algorithms. Each algorithm is assigned an explicit, consistent trajectory color.
                - **Learning Rate (alpha)**: Step scale multiplier. Excessive values induce numerical overflow; undersized values stall progression.
                - **Max Steps**: Total execution budget before halting simulation.
                - **Initial Coordinates (x0, y0)**: Coordinates where algorithms begin traversal.
                - **Gradient Field (Quivers)**: Overlays normalized vectors indicating local steepest descent direction $-\\nabla f(x, y)$.
                - **Logarithmic Contours**: Compresses high-magnitude cliff regions so valley floor geometry remains discernible.
                """
            )

        with col_g2:
            st.markdown(
                """
                ### Graph Interaction Guide
                - **Zooming**: Scroll mouse wheel over any chart to zoom at cursor position, or click-and-drag a box to zoom into a region.
                - **Reset View**: Double-click anywhere on the canvas to restore default bounds.
                - **Panning**: Select the pan tool from the top-right toolbar or click-and-drag.
                - **3D Manifold Rotation**: Click and drag on the 3D surface plot to orbit from any vantage point.
                - **Camera Persistence**: Zoom levels and 3D camera angles remain fixed across hyperparameter updates.
                - **Marker Legend**:
                  - **Yellow Star**: Analytical global minimum $(x^*, y^*)$.
                  - **Circle Marker**: Initial starting position $(x_0, y_0)$.
                  - **Cross Marker**: Converged state ($\\|\\nabla f\\| < 10^{-6}$).
                  - **Diamond Marker**: Terminated upon reaching maximum step budget.
                  - **X Marker**: Diverged (numerical overflow or unbounded coordinates).
                """
            )

        st.markdown("---")
        st.subheader("Canonical Test Scenarios")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown(
                """
                **1. Rosenbrock Valley (Curved Ravine)**
                - **Surface**: Rosenbrock
                - **Dynamics**: Vanilla SGD oscillates perpendicularly against steep valley walls. Momentum and Adam dampen orthogonal oscillations and advance down the curved floor toward $(1, 1)$.
                """
            )
        with col_e2:
            st.markdown(
                """
                **2. Saddle Point Escape (Zero Gradient)**
                - **Surface**: Escapable Saddle
                - **Dynamics**: Initialization near $(0, 0)$ stalls vanilla SGD due to vanishing gradient. Momentum and Adam build velocity along negative curvature to reach the global minima at $(0, \\pm\\sqrt{2})$.
                """
            )
        with col_e3:
            st.markdown(
                """
                **3. Ill-Conditioned Quadratic (Anisotropy)**
                - **Surface**: Quadratic Bowl ($\kappa=20$)
                - **Dynamics**: The Hessian condition number causes high gradient variance across axes. Momentum cancels cross-axis bouncing to accelerate along the primary axis.
                """
            )

    with tab_2d:
        st.subheader(f"2D Contour Map: {surface.name}")
        fig_2d = plot_2d_contour(
            surface=surface,
            results=results,
            grid_res=140,
            show_vector_field=show_vector_field,
            log_scale_contours=log_scale_contours,
        )
        st.plotly_chart(fig_2d, use_container_width=True, config=PLOT_CONFIG)
        st.caption("Note: Scroll mouse wheel to zoom at cursor position. Double-click to reset view.")

    with tab_3d:
        st.subheader(f"3D Loss Landscape: {surface.name}")
        fig_3d = plot_3d_surface(surface=surface, results=results, grid_res=65)
        st.plotly_chart(fig_3d, use_container_width=True, config=PLOT_CONFIG)
        st.caption("Note: Click and drag to orbit the 3D landscape. Scroll to zoom.")

    with tab_convergence:
        st.subheader("Trajectory Convergence Diagnostics")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_loss = plot_loss_curves(results)
            st.plotly_chart(fig_loss, use_container_width=True, config=PLOT_CONFIG)
        with col_c2:
            fig_dist = plot_distance_curves(results)
            st.plotly_chart(fig_dist, use_container_width=True, config=PLOT_CONFIG)
        st.caption("Note: Hover along horizontal axis to inspect all active optimizer values simultaneously.")

    with tab_metrics:
        st.subheader("Performance Summary Table")
        table_rows = []
        for name, res in results.items():
            status = "Converged" if res.converged else ("Diverged" if res.diverged else "Max Steps Reached")
            last_p = res.points[-1] if res.points else None
            table_rows.append(
                {
                    "Optimizer": name,
                    "Status": status,
                    "Steps": res.steps_taken,
                    "Final Loss": f"{res.final_loss:.5e}",
                    "Min Loss": f"{res.min_loss:.5e}",
                    "Final ||∇f||": f"{last_p.grad_norm:.5e}" if last_p else "-",
                    "Final Dist to Min": f"{last_p.distance_to_min:.4f}" if last_p else "-",
                    "Termination Reason": res.termination_reason,
                }
            )

        df_metrics = pd.DataFrame(table_rows)
        st.dataframe(df_metrics, use_container_width=True)

        # Export trajectory data
        csv_rows = []
        for name, res in results.items():
            for p in res.points:
                csv_rows.append(
                    {
                        "optimizer": name,
                        "step": p.step,
                        "x": p.x,
                        "y": p.y,
                        "loss": p.loss,
                        "grad_norm": p.grad_norm,
                        "dist_to_min": p.distance_to_min,
                    }
                )
        df_csv = pd.DataFrame(csv_rows)
        st.download_button(
            label="Export Trajectory Data (CSV)",
            data=df_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"trajectories_{selected_surface_id}.csv",
            mime="text/csv",
        )

    with tab_mechanics:
        st.subheader("Optimizer Formulations and Mathematical Mechanics")
        st.markdown(
            r"""
            ### 1. Vanilla SGD (Stochastic / Standard Gradient Descent)
            $$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$
            - **Mechanics**: Updates parameters strictly along instantaneous negative gradient. In ill-conditioned ravines, orthogonal components dominate, leading to high oscillation and slow longitudinal progress.

            ### 2. Polyak Momentum
            $$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
            - **Mechanics**: Implements physical momentum via velocity vector $v_t$. High-frequency perpendicular oscillations cancel out, while low-frequency longitudinal signals accumulate, accelerating convergence along ravines.

            ### 3. Nesterov Accelerated Gradient (NAG)
            $$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t - \beta v_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
            - **Mechanics**: Evaluates gradient at an anticipated lookahead position $(\theta_t - \beta v_t)$. Provides predictive braking that reduces overshoot near valley bottoms.

            ### 4. AdaGrad (Adaptive Gradient)
            $$G_{t+1} = G_t + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_{t+1}} + \epsilon} \odot g_t$$
            - **Mechanics**: Scales learning rates element-wise according to cumulative sum of squared gradients. Useful for sparse features, but monotonic growth of $G_t$ prematurely decays effective learning rate to zero on non-convex plateaus.

            ### 5. RMSprop (Root Mean Square Propagation)
            $$v_{t+1} = \gamma v_t + (1 - \gamma) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_{t+1}} + \epsilon} \odot g_t$$
            - **Mechanics**: Replaces cumulative sum with an Exponential Moving Average (EMA). Constrains memory of past squared gradients, preventing premature step freezing.

            ### 6. Adam (Adaptive Moment Estimation)
            $$\hat{m}_{t+1} = \frac{\beta_1 m_t + (1 - \beta_1) g_t}{1 - \beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{\beta_2 v_t + (1 - \beta_2) g_t^2}{1 - \beta_2^{t+1}}$$
            $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$$
            - **Mechanics**: Integrates first-moment velocity with second-moment adaptive scaling, supplemented by initialization bias corrections for early iterations.
            """
        )


if __name__ == "__main__":
    main()
