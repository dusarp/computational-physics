# Diffusion Equation Solver

A comprehensive Python implementation for solving the 1D and 2D diffusion equation using finite difference methods, with visualization and parameter experiments.

## Overview

This project implements numerical solvers for the diffusion equation (also known as the heat equation):

**1D Diffusion Equation:**
```
∂u/∂t = D * ∂²u/∂x²
```

**2D Diffusion Equation:**
```
∂u/∂t = D * (∂²u/∂x² + ∂²u/∂y²)
```

where:
- `u(x,t)` or `u(x,y,t)` is the quantity being diffused (e.g., temperature, concentration)
- `D` is the diffusion coefficient
- `t` is time
- `x, y` are spatial coordinates

## Mathematical Method

The solvers use the **explicit finite difference method** (Forward-Time Central-Space or FTCS scheme):

### 1D Discretization
```
u[i]^(n+1) = u[i]^n + α(u[i+1]^n - 2u[i]^n + u[i-1]^n)
```

where `α = D*dt/dx²`

### 2D Discretization
```
u[i,j]^(n+1) = u[i,j]^n + α_x(u[i,j+1]^n - 2u[i,j]^n + u[i,j-1]^n) 
                         + α_y(u[i+1,j]^n - 2u[i,j]^n + u[i-1,j]^n)
```

where `α_x = D*dt/dx²` and `α_y = D*dt/dy²`

### Stability Condition

For numerical stability, the method requires:
- **1D:** `α ≤ 0.5`
- **2D:** `α_x + α_y ≤ 0.5`

The solver automatically checks this condition and warns if violated.

## Features

- ✅ 1D and 2D diffusion solvers
- ✅ Multiple initial condition options:
  - Gaussian distributions
  - Step functions
  - Sine waves
  - Circular regions (2D)
  - Double Gaussians (2D)
- ✅ Automatic stability checking
- ✅ Time evolution visualization
- ✅ 3D surface plots for 2D solutions
- ✅ Parameter experiments
- ✅ Publication-quality plots

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Seaborn

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete experiment suite:

```bash
python diffusion_solver.py
```

This will generate all experimental results and save visualizations to the `outputs/` directory.

### Custom Usage

#### 1D Diffusion Example

```python
from diffusion_solver import DiffusionSolver1D

# Initialize solver
solver = DiffusionSolver1D(
    L=1.0,      # Domain length
    nx=100,     # Number of grid points
    D=0.1,      # Diffusion coefficient
    dt=0.0001   # Time step
)

# Set initial condition (Gaussian)
solver.set_initial_condition(solver.gaussian_ic(x0=0.5, sigma=0.05))

# Solve the equation
solver.solve(t_final=1.0, save_interval=10)

# Visualize the results
fig = solver.plot_evolution(n_snapshots=5)
plt.show()
```

#### 2D Diffusion Example

```python
from diffusion_solver import DiffusionSolver2D

# Initialize 2D solver
solver_2d = DiffusionSolver2D(
    Lx=1.0, Ly=1.0,  # Domain dimensions
    nx=50, ny=50,     # Grid points
    D=0.05,           # Diffusion coefficient
    dt=0.0001         # Time step
)

# Set initial condition (Gaussian)
solver_2d.set_initial_condition(
    solver_2d.gaussian_ic(x0=0.5, y0=0.5, sigma=0.1)
)

# Solve
solver_2d.solve(t_final=0.5, save_interval=50)

# Visualize as contour plots
fig = solver_2d.plot_evolution(n_snapshots=4)
plt.show()

# Or as 3D surface
fig_3d = solver_2d.plot_3d_surface(time_idx=-1)
plt.show()
```

## Parameter Experiments

The script includes 6 comprehensive experiments:

### Experiment 1: Diffusion Coefficient Effects
Compares solutions with different diffusion coefficients (D = 0.01, 0.05, 0.1):
- Higher D → faster diffusion
- Lower D → slower spreading

### Experiment 2: Initial Conditions
Tests three different initial conditions:
- **Gaussian**: Smooth bell curve
- **Step function**: Sharp discontinuity
- **Sine wave**: Oscillatory pattern

### Experiment 3: 2D Gaussian Diffusion
Demonstrates diffusion of a 2D Gaussian peak over time

### Experiment 4: Multiple Sources (2D)
Shows interaction between two Gaussian sources as they diffuse and merge

### Experiment 5: 3D Visualization
3D surface plot showing the final state of a 2D diffusion process

### Experiment 6: Grid Resolution
Compares numerical accuracy with different spatial resolutions (nx = 50, 100, 200)

## Output Files

Running the experiments generates the following visualizations:

```
outputs/
├── experiment_1_diffusion_coefficients.png
├── experiment_2_initial_conditions.png
├── experiment_3_2d_gaussian.png
├── experiment_4_2d_double_gaussian.png
├── experiment_5_3d_surface.png
└── experiment_6_grid_resolution.png
```

## Physical Interpretation

The diffusion equation models many physical phenomena:

- **Heat conduction**: Temperature distribution in a rod or plate
- **Mass diffusion**: Concentration of particles in a fluid
- **Chemical reactions**: Spreading of chemical species
- **Population dynamics**: Dispersal of biological populations

### Key Parameters

- **D (Diffusion coefficient)**:
  - Units: [length²/time]
  - Physical meaning: How quickly the quantity spreads
  - Typical values: 
    - Thermal diffusivity of metals: ~10⁻⁵ m²/s
    - Molecular diffusion in liquids: ~10⁻⁹ m²/s

- **Boundary conditions**:
  - Currently implements Dirichlet (fixed value at boundaries)
  - Set to u=0 at all boundaries

## Numerical Considerations

### Choosing Parameters

1. **Spatial resolution (dx, dy)**:
   - Smaller → more accurate but slower
   - Should be fine enough to resolve initial condition features

2. **Time step (dt)**:
   - Must satisfy stability condition
   - Smaller is safer but requires more iterations

3. **Recommended approach**:
   ```python
   dx = L / (nx - 1)
   dt = 0.4 * dx**2 / D  # Sets α = 0.4, safely below 0.5
   ```

### Performance Tips

- For large 2D grids, consider using vectorized NumPy operations
- Use `save_interval` to avoid storing every time step
- Balance `nx/ny` and `dt` for optimal accuracy vs. speed

## Extension Ideas

Potential enhancements:

1. **Different boundary conditions**:
   - Neumann (insulated boundaries)
   - Periodic boundaries
   - Mixed conditions

2. **Implicit methods**:
   - Crank-Nicolson scheme (unconditionally stable)
   - Backward Euler

3. **Nonlinear diffusion**:
   - Concentration-dependent D
   - Reaction-diffusion equations

4. **3D solver**:
   - Extend to three spatial dimensions

5. **Adaptive time stepping**:
   - Automatically adjust dt for efficiency

## Mathematical Background

The diffusion equation is a parabolic partial differential equation derived from conservation laws:

```
∂u/∂t + ∇·J = 0
```

with Fick's law: `J = -D∇u`

This gives: `∂u/∂t = D∇²u`

**Analytical solutions** exist for simple cases:
- 1D with Gaussian IC: `u(x,t) = (1/√(4πDt)) * exp(-x²/(4Dt))`

## References

1. **Numerical Methods**:
   - LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*

2. **Diffusion Theory**:
   - Crank, J. (1975). *The Mathematics of Diffusion*

3. **Physical Applications**:
   - Incropera, F. P., & DeWitt, D. P. (2002). *Fundamentals of Heat and Mass Transfer*

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new initial conditions
- Implement different numerical schemes

## Author

Created for educational and research purposes.

## Citation

If you use this code in your research, please cite:
```
Diffusion Equation Solver (2015)

```