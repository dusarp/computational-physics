"""
Diffusion Equation Solver
Solves the 1D and 2D diffusion equation using finite difference methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)


class DiffusionSolver1D:
    """
    Solves the 1D diffusion equation: ∂u/∂t = D * ∂²u/∂x²
    using the explicit finite difference method (FTCS scheme)
    """
    
    def __init__(self, L=1.0, nx=100, D=0.1, dt=0.0001):
        """
        Parameters:
        -----------
        L : float
            Length of the domain
        nx : int
            Number of spatial grid points
        D : float
            Diffusion coefficient
        dt : float
            Time step
        """
        self.L = L
        self.nx = nx
        self.D = D
        self.dt = dt
        
        # Spatial grid
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        # Stability condition (CFL condition)
        self.alpha = D * dt / (self.dx ** 2)
        if self.alpha > 0.5:
            print(f"Warning: Stability condition violated! α = {self.alpha:.3f} > 0.5")
            print(f"Consider reducing dt or increasing dx")
        
        # Initialize solution
        self.u = None
        self.u_history = []
        
    def set_initial_condition(self, u_initial):
        """Set initial condition u(x, 0) = u_initial"""
        if callable(u_initial):
            self.u = u_initial(self.x)
        else:
            self.u = u_initial.copy()
        self.u_history = [self.u.copy()]
        
    def gaussian_ic(self, x0=0.5, sigma=0.05):
        """Gaussian initial condition"""
        return np.exp(-((self.x - x0) ** 2) / (2 * sigma ** 2))
    
    def step_ic(self, x_left=0.4, x_right=0.6):
        """Step function initial condition"""
        u = np.zeros(self.nx)
        u[(self.x >= x_left) & (self.x <= x_right)] = 1.0
        return u
    
    def sine_ic(self, n=1):
        """Sine wave initial condition"""
        return np.sin(n * np.pi * self.x / self.L)
    
    def solve(self, t_final=1.0, save_interval=10):
        """
        Solve the diffusion equation from t=0 to t=t_final
        
        Parameters:
        -----------
        t_final : float
            Final time
        save_interval : int
            Save solution every save_interval steps
        """
        if self.u is None:
            raise ValueError("Initial condition not set. Use set_initial_condition()")
        
        n_steps = int(t_final / self.dt)
        
        for step in range(n_steps):
            u_new = self.u.copy()
            
            # Update interior points using FTCS scheme
            for i in range(1, self.nx - 1):
                u_new[i] = self.u[i] + self.alpha * (
                    self.u[i+1] - 2*self.u[i] + self.u[i-1]
                )
            
            # Boundary conditions (Dirichlet: u=0 at boundaries)
            u_new[0] = 0
            u_new[-1] = 0
            
            self.u = u_new
            
            # Save solution at specified intervals
            if step % save_interval == 0:
                self.u_history.append(self.u.copy())
        
        return self.u
    
    def plot_evolution(self, n_snapshots=5):
        """Plot the evolution of the solution at different time steps"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_total = len(self.u_history)
        indices = np.linspace(0, n_total-1, n_snapshots, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        
        for idx, color in zip(indices, colors):
            time = idx * len(self.u_history) * self.dt
            ax.plot(self.x, self.u_history[idx], color=color, 
                   label=f't = {time:.4f}', linewidth=2)
        
        ax.set_xlabel('Position x', fontsize=12)
        ax.set_ylabel('u(x, t)', fontsize=12)
        ax.set_title(f'1D Diffusion Equation (D = {self.D})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class DiffusionSolver2D:
    """
    Solves the 2D diffusion equation: ∂u/∂t = D * (∂²u/∂x² + ∂²u/∂y²)
    using the explicit finite difference method
    """
    
    def __init__(self, Lx=1.0, Ly=1.0, nx=50, ny=50, D=0.1, dt=0.0001):
        """
        Parameters:
        -----------
        Lx, Ly : float
            Domain dimensions
        nx, ny : int
            Number of spatial grid points
        D : float
            Diffusion coefficient
        dt : float
            Time step
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.D = D
        self.dt = dt
        
        # Spatial grids
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Stability condition
        self.alpha_x = D * dt / (self.dx ** 2)
        self.alpha_y = D * dt / (self.dy ** 2)
        total_alpha = self.alpha_x + self.alpha_y
        
        if total_alpha > 0.5:
            print(f"Warning: Stability condition violated! α_total = {total_alpha:.3f} > 0.5")
            print(f"Consider reducing dt or increasing dx/dy")
        
        # Initialize solution
        self.u = None
        self.u_history = []
        
    def set_initial_condition(self, u_initial):
        """Set initial condition u(x, y, 0) = u_initial"""
        if callable(u_initial):
            self.u = u_initial(self.X, self.Y)
        else:
            self.u = u_initial.copy()
        self.u_history = [self.u.copy()]
        
    def gaussian_ic(self, x0=0.5, y0=0.5, sigma=0.1):
        """2D Gaussian initial condition"""
        return np.exp(-(((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2 * sigma ** 2)))
    
    def double_gaussian_ic(self):
        """Two Gaussian peaks"""
        g1 = np.exp(-(((self.X - 0.3) ** 2 + (self.Y - 0.3) ** 2) / (2 * 0.05 ** 2)))
        g2 = np.exp(-(((self.X - 0.7) ** 2 + (self.Y - 0.7) ** 2) / (2 * 0.05 ** 2)))
        return g1 + g2
    
    def circle_ic(self, x0=0.5, y0=0.5, radius=0.2):
        """Circular initial condition"""
        u = np.zeros_like(self.X)
        mask = ((self.X - x0) ** 2 + (self.Y - y0) ** 2) <= radius ** 2
        u[mask] = 1.0
        return u
    
    def solve(self, t_final=1.0, save_interval=10):
        """
        Solve the 2D diffusion equation from t=0 to t=t_final
        
        Parameters:
        -----------
        t_final : float
            Final time
        save_interval : int
            Save solution every save_interval steps
        """
        if self.u is None:
            raise ValueError("Initial condition not set. Use set_initial_condition()")
        
        n_steps = int(t_final / self.dt)
        
        for step in range(n_steps):
            u_new = self.u.copy()
            
            # Update interior points
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    u_new[i, j] = self.u[i, j] + self.alpha_x * (
                        self.u[i, j+1] - 2*self.u[i, j] + self.u[i, j-1]
                    ) + self.alpha_y * (
                        self.u[i+1, j] - 2*self.u[i, j] + self.u[i-1, j]
                    )
            
            # Boundary conditions (Dirichlet: u=0 at all boundaries)
            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0
            
            self.u = u_new
            
            # Save solution at specified intervals
            if step % save_interval == 0:
                self.u_history.append(self.u.copy())
        
        return self.u
    
    def plot_evolution(self, n_snapshots=4):
        """Plot the evolution of the 2D solution"""
        n_total = len(self.u_history)
        indices = np.linspace(0, n_total-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, ax in zip(indices, axes):
            time = idx * len(self.u_history) * self.dt
            
            im = ax.contourf(self.X, self.Y, self.u_history[idx], 
                           levels=20, cmap='hot')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            ax.set_title(f't = {time:.4f}', fontsize=12)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(f'2D Diffusion Equation (D = {self.D})', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_3d_surface(self, time_idx=-1):
        """Plot 3D surface of the solution at a specific time"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self.X, self.Y, self.u_history[time_idx], 
                              cmap='hot', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_zlabel('u(x, y, t)', fontsize=10)
        
        time = time_idx * self.dt * len(self.u_history)
        ax.set_title(f'2D Diffusion at t = {time:.4f}', fontsize=12)
        
        fig.colorbar(surf, ax=ax, shrink=0.5)
        
        return fig


def run_experiments():
    """Run various experiments with different parameters"""
    
    print("=" * 60)
    print("DIFFUSION EQUATION SOLVER - PARAMETER EXPERIMENTS")
    print("=" * 60)
    
    # Experiment 1: 1D Diffusion with different diffusion coefficients
    print("\n1. Testing different diffusion coefficients (1D)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    D_values = [0.01, 0.05, 0.1]
    
    for ax, D in zip(axes, D_values):
        solver = DiffusionSolver1D(L=1.0, nx=100, D=D, dt=0.0001)
        solver.set_initial_condition(solver.gaussian_ic())
        solver.solve(t_final=0.5, save_interval=50)
        
        n_snapshots = 5
        n_total = len(solver.u_history)
        indices = np.linspace(0, n_total-1, n_snapshots, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        
        for idx, color in zip(indices, colors):
            time = idx * 50 * solver.dt
            ax.plot(solver.x, solver.u_history[idx], color=color, 
                   label=f't={time:.3f}', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f'D = {D}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Effect of Diffusion Coefficient (1D)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment_1_diffusion_coefficients.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_1_diffusion_coefficients.png")
    
    # Experiment 2: 1D Diffusion with different initial conditions
    print("\n2. Testing different initial conditions (1D)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Gaussian IC
    solver1 = DiffusionSolver1D(L=1.0, nx=100, D=0.05, dt=0.0001)
    solver1.set_initial_condition(solver1.gaussian_ic())
    solver1.solve(t_final=0.5, save_interval=100)
    solver1.plot_evolution(n_snapshots=5)
    axes[0].clear()
    
    n_snapshots = 5
    n_total = len(solver1.u_history)
    indices = np.linspace(0, n_total-1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    for idx, color in zip(indices, colors):
        time = idx * 100 * solver1.dt
        axes[0].plot(solver1.x, solver1.u_history[idx], color=color, 
                    label=f't={time:.3f}', linewidth=2)
    axes[0].set_title('Gaussian IC')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x, t)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Step function IC
    solver2 = DiffusionSolver1D(L=1.0, nx=100, D=0.05, dt=0.0001)
    solver2.set_initial_condition(solver2.step_ic())
    solver2.solve(t_final=0.5, save_interval=100)
    
    for idx, color in zip(indices, colors):
        time = idx * 100 * solver2.dt
        axes[1].plot(solver2.x, solver2.u_history[idx], color=color, 
                    label=f't={time:.3f}', linewidth=2)
    axes[1].set_title('Step Function IC')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x, t)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Sine wave IC
    solver3 = DiffusionSolver1D(L=1.0, nx=100, D=0.05, dt=0.0001)
    solver3.set_initial_condition(solver3.sine_ic(n=2))
    solver3.solve(t_final=0.5, save_interval=100)
    
    for idx, color in zip(indices, colors):
        time = idx * 100 * solver3.dt
        axes[2].plot(solver3.x, solver3.u_history[idx], color=color, 
                    label=f't={time:.3f}', linewidth=2)
    axes[2].set_title('Sine Wave IC')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('u(x, t)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Different Initial Conditions (1D, D=0.05)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment_2_initial_conditions.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_2_initial_conditions.png")
    
    # Experiment 3: 2D Diffusion with Gaussian initial condition
    print("\n3. Testing 2D diffusion with Gaussian IC...")
    
    solver_2d = DiffusionSolver2D(Lx=1.0, Ly=1.0, nx=50, ny=50, D=0.05, dt=0.0001)
    solver_2d.set_initial_condition(solver_2d.gaussian_ic())
    solver_2d.solve(t_final=0.5, save_interval=100)
    fig = solver_2d.plot_evolution(n_snapshots=4)
    plt.savefig('/mnt/user-data/outputs/experiment_3_2d_gaussian.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_3_2d_gaussian.png")
    
    # Experiment 4: 2D Diffusion with double Gaussian
    print("\n4. Testing 2D diffusion with double Gaussian IC...")
    
    solver_2d_double = DiffusionSolver2D(Lx=1.0, Ly=1.0, nx=50, ny=50, D=0.05, dt=0.0001)
    solver_2d_double.set_initial_condition(solver_2d_double.double_gaussian_ic())
    solver_2d_double.solve(t_final=0.5, save_interval=100)
    fig = solver_2d_double.plot_evolution(n_snapshots=4)
    plt.savefig('/mnt/user-data/outputs/experiment_4_2d_double_gaussian.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_4_2d_double_gaussian.png")
    
    # Experiment 5: 2D Diffusion 3D surface plot
    print("\n5. Creating 3D surface plot for 2D diffusion...")
    
    fig = solver_2d.plot_3d_surface(time_idx=-1)
    plt.savefig('/mnt/user-data/outputs/experiment_5_3d_surface.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_5_3d_surface.png")
    
    # Experiment 6: Grid resolution comparison (1D)
    print("\n6. Testing different grid resolutions (1D)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nx_values = [50, 100, 200]
    
    for ax, nx in zip(axes, nx_values):
        # Adjust dt to maintain stability
        dx = 1.0 / (nx - 1)
        dt = 0.4 * dx**2 / 0.05  # Keep alpha = 0.4
        
        solver = DiffusionSolver1D(L=1.0, nx=nx, D=0.05, dt=dt)
        solver.set_initial_condition(solver.gaussian_ic())
        
        # Adjust save_interval to have similar number of snapshots
        save_interval = max(1, int(0.5 / dt / 50))
        solver.solve(t_final=0.5, save_interval=save_interval)
        
        n_snapshots = 5
        n_total = len(solver.u_history)
        indices = np.linspace(0, n_total-1, n_snapshots, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        
        for idx, color in zip(indices, colors):
            time = idx * save_interval * dt
            ax.plot(solver.x, solver.u_history[idx], color=color, 
                   label=f't={time:.3f}', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f'nx = {nx} (dx = {dx:.4f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Effect of Grid Resolution (1D, D=0.05)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment_6_grid_resolution.png', dpi=150, bbox_inches='tight')
    print("   Saved: experiment_6_grid_resolution.png")
    
    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print("All figures saved to /mnt/user-data/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    run_experiments()