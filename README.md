# Gradient Descent Visualizer

An interactive simulation tool to analyze and compare first-order optimization algorithms (SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam) across non-convex and ill-conditioned loss surfaces.

Built with Python, NumPy, Plotly, and Streamlit.

---

## Motivation

Most machine learning libraries treat optimizers like Adam or RMSprop as plug-and-play black boxes. In practice, the local geometry of the loss surface (such as high condition number ravines, flat plateaus, or saddle points) dictates whether an algorithm converges, oscillates, or diverges.

I built this project to study and visualize:
1. **Ill-conditioned ravines**: Why vanilla gradient descent oscillates across steep valley walls without making forward progress, and how momentum and adaptive learning rates fix this.
2. **Saddle points**: How momentum-based methods build velocity to escape zero-gradient saddle points where standard SGD stalls.
3. **Adaptive learning rates**: The practical differences between coordinate-wise accumulators (AdaGrad) and exponential moving averages (RMSprop, Adam).

---

## Features

- **Benchmark Loss Landscapes**: Analytical gradients for 7 non-convex test surfaces (Rosenbrock, Beale, Rastrigin, Saddle Point, Himmelblau, Ill-conditioned Quadratic Bowl, and Sphere).
- **First-Order Optimizers**: Implementations of SGD, Polyak Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSprop, and Adam.
- **Interactive Visualizations**:
  - **2D Contour Plot**: High-resolution contour maps, gradient direction vectors, and colored trajectory paths.
  - **3D Loss Landscape**: Rotatable 3D surface mesh with trajectories plotted directly onto the surface.
  - **Convergence Curves**: Log-scaled loss vs. step count and distance to known global minima.
- **Multi-Optimizer Comparison**: Run multiple algorithms simultaneously from identical starting coordinates to compare path efficiency and speed.
- **CLI and Headless Runs**: Run simulations from the terminal and export interactive HTML reports.
- **Unit Tests**: 27 unit tests validating analytical gradients against central finite difference approximations and checking optimizer convergence.

---

## Implemented Algorithms

| Optimizer | Core Update Rule | Key Hyperparameters | Intuition |
| :--- | :--- | :--- | :--- |
| **SGD** | $\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$ | $\alpha$ (learning rate) | Basic steepest descent. Oscillates in steep ravines. |
| **Momentum** | $v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ | $\alpha, \beta=0.9$ | Heavy ball mechanics. Cancels orthogonal oscillations, builds speed down ravines. |
| **NAG** | $v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t - \beta v_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ | $\alpha, \beta=0.9$ | Lookahead momentum. Anticipatory braking prevents overshooting narrow basins. |
| **AdaGrad** | $G_{t+1} = G_t + g_t^2$<br>$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_{t+1}} + \epsilon} g_t$ | $\alpha, \epsilon=10^{-8}$ | Adapts learning rate per parameter. Can stall prematurely due to monotonic growth of $G_t$. |
| **RMSprop** | $v_{t+1} = \gamma v_t + (1 - \gamma) g_t^2$<br>$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_{t+1}} + \epsilon} g_t$ | $\alpha, \gamma=0.9, \epsilon=10^{-8}$ | Uses Exponential Moving Average (EMA) of squared gradients to keep step sizes responsive. |
| **Adam** | $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \; \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$<br>$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | $\alpha, \beta_1=0.9, \beta_2=0.999$ | Combines momentum with adaptive learning rates and initialization bias correction. |

---

## Benchmark Surfaces

- **Rosenbrock ("Banana") Function**: $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$
  - Valley minimum at $(1, 1)$. Tests whether optimizers can follow a sharp curved ridge without diverging.
- **Beale's Function**: $f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$
  - Global minimum at $(3, 0.5)$. Tests navigation around sharp corners and flat plateaus with vanishing gradients.
- **Escapable Saddle Point**: $f(x, y) = x^2 - y^2 + \frac{1}{4}y^4$
  - Saddle at $(0, 0)$, minima at $(0, \pm\sqrt{2})$. Tests escape velocity along negative curvature directions.
- **Ill-Conditioned Quadratic**: $f(x, y) = \frac{1}{2}(x^2 + 20y^2)$
  - Condition number $\kappa = 20$. Demonstrates how anisotropic Hessian eigenvalues cause vanilla gradient descent to bounce between walls.
- **Himmelblau's Function**: $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$
  - 4 identical global minima separated by ridges. Shows basin selection based on initialization.
- **Rastrigin Function**: $f(x, y) = 20 + (x^2 - 10\cos(2\pi x)) + (y^2 - 10\cos(2\pi y))$
  - Multimodal landscape with many local minima. Demonstrates vulnerability of first-order methods to local traps.

---

## Quickstart

### 1. Set Up Environment

```bash
git clone https://github.com/adem-temam/gradient-descent-visualizer.git
cd gradient-descent-visualizer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web App

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Headless CLI Benchmarks

Compare algorithms directly from the terminal:

```bash
# Benchmark on Rosenbrock
python cli.py --surface rosenbrock --optimizers sgd momentum rmsprop adam --steps 120

# Export interactive HTML contour plot
python cli.py --surface quadratic_bowl --optimizers sgd momentum adam --export-html report.html
```

---

## Testing

Run the automated test suite with pytest:

```bash
pytest -v
```

The test suite covers:
- **Gradient Accuracy**: Central finite difference numerical verification across 5 points per surface.
- **Global Minimum Checks**: Verifies that analytical minima produce near-zero gradient and expected loss.
- **Convergence Verification**: Checks that each optimizer converges to within 5% of starting loss on standard convex objectives.
- **Divergence Handling**: Tests handling of extreme learning rates and invalid hyperparameters.

---

## Project Structure

```text
gradient-descent-visualizer/
├── src/
│   └── gd_visualizer/
│       ├── __init__.py
│       ├── functions.py       # Loss landscapes, analytical gradients, domains, minima
│       ├── optimizers.py      # SGD, Momentum, NAG, AdaGrad, RMSprop, Adam
│       ├── runner.py          # Trajectory simulation engine & metrics collection
│       └── visualizer.py      # Plotly 2D contour & 3D surface plot generators
├── tests/
│   ├── __init__.py
│   ├── test_functions.py      # Finite difference gradient checks
│   ├── test_optimizers.py     # Convergence tests on convex quadratics
│   ├── test_runner.py          # Divergence and simulation checks
│   └── test_visualizer.py      # Plotly figure construction tests
├── app.py                     # Streamlit interactive application
├── cli.py                     # CLI tool for headless benchmarks & HTML export
├── pyproject.toml             # Standard modern packaging metadata
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Standard Python gitignore
└── README.md                  # Project documentation
```
