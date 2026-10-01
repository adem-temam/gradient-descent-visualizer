"""Gradient Descent Visualizer - Interactive Streamlit Application.

An interactive dashboard for comparing first-order optimization algorithms
across classic non-convex and ill-conditioned loss surfaces.
"""

from __future__ import annotations

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
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, professional look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("📉 Gradient Descent Visualizer")
    st.caption(
        "Interactive analysis and benchmarking of first-order optimization algorithms on non-convex loss surfaces."
    )

    # --- Sidebar Configuration ---
    st.sidebar.header("🎯 Loss Surface")
    surface_options = {s_id: s.name for s_id, s in SURFACES.items()}
    selected_surface_id = st.sidebar.selectbox(
        "Select Landscape",
        options=list(surface_options.keys()),
        format_func=lambda x: surface_options[x],
        index=0,
    )
    surface = get_surface(selected_surface_id)

    st.sidebar.markdown(f"**Formula:**")
    st.sidebar.latex(surface.formula_latex)
    st.sidebar.info(surface.description)

    st.sidebar.header("⚙️ Optimization Settings")

    # Optimizer Multi-selection
    all_opt_keys = list(OPTIMIZER_REGISTRY.keys())
    selected_opts = st.sidebar.multiselect(
        "Active Optimizers",
        options=all_opt_keys,
        default=["sgd", "momentum", "rmsprop", "adam"],
        format_func=lambda x: OPTIMIZER_REGISTRY[x]().name,
    )

    if not selected_opts:
        st.warning("Please select at least one optimizer from the sidebar.")
        return

    # Hyperparameters
    col_steps, col_lr = st.sidebar.columns(2)
    with col_steps:
        max_steps = st.slider("Max Steps", min_value=10, max_value=400, value=120, step=10)
    with col_lr:
        custom_lr = st.number_input(
            "Global LR",
            min_value=0.0001,
            max_value=2.0,
            value=0.01 if selected_surface_id != "beale" else 0.001,
            step=0.005,
            format="%.4f",
        )

    # Starting coordinates
    st.sidebar.subheader("📍 Starting Position (x₀, y₀)")
    col_x, col_y = st.sidebar.columns(2)
    x_min, x_max = surface.domain_x
    y_min, y_max = surface.domain_y

    with col_x:
        start_x = st.slider(
            "x₀",
            min_value=float(x_min),
            max_value=float(x_max),
            value=float(surface.default_start[0]),
            step=0.05,
        )
    with col_y:
        start_y = st.slider(
            "y₀",
            min_value=float(y_min),
            max_value=float(y_max),
            value=float(surface.default_start[1]),
            step=0.05,
        )

    # Visualization toggles
    st.sidebar.subheader("🎨 Display Options")
    show_vector_field = st.sidebar.checkbox("Show Gradient Field (Quivers)", value=False)
    log_scale_contours = st.sidebar.checkbox("Logarithmic Contours", value=True)

    # Run Benchmark Simulation
    optimizers = [create_optimizer(opt_id, lr=custom_lr) for opt_id in selected_opts]
    results = run_benchmark(
        surface=surface,
        optimizers=optimizers,
        start_pos=(start_x, start_y),
        max_steps=max_steps,
    )

    # --- Main Content Tabs ---
    tab_2d, tab_3d, tab_convergence, tab_metrics, tab_guide = st.tabs(
        [
            "🗺️ 2D Contour & Trajectories",
            "🌐 3D Loss Landscape",
            "📈 Convergence Curves",
            "📊 Benchmark Metrics",
            "📖 Algorithm Mechanics",
        ]
    )

    with tab_2d:
        fig_2d = plot_2d_contour(
            surface=surface,
            results=results,
            grid_res=140,
            show_vector_field=show_vector_field,
            log_scale_contours=log_scale_contours,
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab_3d:
        fig_3d = plot_3d_surface(surface=surface, results=results, grid_res=65)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_convergence:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_loss = plot_loss_curves(results)
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_c2:
            fig_dist = plot_distance_curves(results)
            st.plotly_chart(fig_dist, use_container_width=True)

    with tab_metrics:
        st.subheader("Performance Summary")
        table_rows = []
        for name, res in results.items():
            status = "✅ Converged" if res.converged else ("⚠️ Diverged" if res.diverged else "⏱️ Max Steps")
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

        # Download trajectory data as CSV
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
            label="📥 Download Trajectory CSV Data",
            data=df_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"trajectories_{selected_surface_id}.csv",
            mime="text/csv",
        )

    with tab_guide:
        st.subheader("Optimizer Formulations & Intuitions")
        st.markdown(
            r"""
            ### 1. Vanilla SGD
            $$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$
            - **Behavior**: Relies purely on instantaneous local gradient. In ill-conditioned valleys (e.g. Rosenbrock or elongated quadratics), the gradient points almost orthogonally across the valley walls rather than along the floor, leading to violent oscillations and slow forward progress.

            ### 2. Polyak Momentum
            $$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
            - **Behavior**: Accumulates past velocity like a physical heavy ball rolling downhill. Oscillations across narrow walls cancel out, while momentum builds in consistent directions, accelerating traversal through ravines. Can overshoot sharp corners.

            ### 3. Nesterov Accelerated Gradient (NAG)
            $$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t - \beta v_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
            - **Behavior**: Computes gradient at an anticipated lookahead position ($\theta_t - \beta v_t$). Acts as an adaptive braking mechanism when approaching sharp turns or bottoms of valleys.

            ### 4. AdaGrad
            $$G_{t+1} = G_t + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_{t+1}} + \epsilon} \odot g_t$$
            - **Behavior**: Adapts learning rates coordinate-wise based on historical gradient magnitudes. High-frequency dimensions receive smaller updates, while infrequent dimensions receive larger updates. Drawback: Monotonically increasing accumulator shrinks step sizes to zero prematurely.

            ### 5. RMSprop
            $$v_{t+1} = \gamma v_t + (1 - \gamma) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_{t+1}} + \epsilon} \odot g_t$$
            - **Behavior**: Fixes AdaGrad's premature freezing by replacing cumulative sums with an Exponential Moving Average (EMA). Steps remain responsive throughout long optimization trajectories.

            ### 6. Adam (Adaptive Moment Estimation)
            $$\hat{m}_{t+1} = \frac{\beta_1 m_t + (1 - \beta_1) g_t}{1 - \beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{\beta_2 v_t + (1 - \beta_2) g_t^2}{1 - \beta_2^{t+1}}$$
            $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$$
            - **Behavior**: Combines the directional velocity of momentum with coordinate-wise adaptive scaling and early-step bias correction. Robust across a wide variety of non-convex geometries.
            """
        )


if __name__ == "__main__":
    main()
