"""
Figure 1: Topological Discrimination Between Ordered and Chaotic Dynamics
RCDT Hypothesis - Aim 1 Calibration (Instrument Validation)

Generates persistence diagrams for:
- Panel A/B: Van der Pol oscillator (limit cycle, ordered dynamics)
- Panel C/D: Lorenz system (chaos, complex dynamics)

Uses: Takens time-delay embedding + Vietoris-Rips filtration (via Ripser)
Output: 2x2 composite figure suitable for bioRxiv preprint
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ripser

try:
    from persim import plot_diagrams
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

# =============================================================================
# 1. Dynamical Systems
# =============================================================================

def van_der_pol(t, y, mu=1.0):
    """Van der Pol oscillator: d²x/dt² - mu(1-x²)dx/dt + x = 0."""
    x, v = y
    return [v, mu * (1 - x**2) * v - x]

def lorenz(t, y, sigma=10.0, rho=28.0, beta=8/3):
    """Lorenz system: dx/dt = sigma(y-x), dy/dt = x(rho-z)-y, dz/dt = xy - beta*z."""
    x, y_, z = y
    return [sigma * (y_ - x), x * (rho - z) - y_, x * y_ - beta * z]

# =============================================================================
# 2. Time Series Generation
# =============================================================================

def generate_time_series(system_fn, y0, t_span, t_eval, obs_idx=0):
    """
    Integrate ODE and return single scalar observable (simulating experimental conditions).
    """
    sol = solve_ivp(system_fn, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
    x = sol.y[obs_idx]  # scalar observable
    return x

# =============================================================================
# 3. Takens Time-Delay Embedding
# =============================================================================

def takens_embedding(x, m=3, tau=1):
    """
    Reconstruct state space from scalar time series via Takens embedding.
    
    X(t) = [x(t), x(t+tau), x(t+2*tau), ..., x(t+(m-1)*tau)]
    
    Parameters:
    -----------
    x : 1D array, scalar time series
    m : int, embedding dimension
    tau : int, embedding delay (in samples)
    
    Returns:
    --------
    X : 2D array of shape (N - (m-1)*tau, m)
    """
    n = len(x)
    N_embed = n - (m - 1) * tau
    X = np.zeros((N_embed, m))
    for i in range(m):
        X[:, i] = x[i * tau : i * tau + N_embed]
    return X

# =============================================================================
# 4. Persistence Diagram Computation
# =============================================================================

def compute_persistence(point_cloud, maxdim=1):
    """
    Compute persistence diagrams using Vietoris-Rips filtration (Ripser).
    
    Returns:
    --------
    diagrams : list of (birth, death) arrays for H0, H1, ...
    """
    result = ripser.ripser(point_cloud, maxdim=maxdim)
    diagrams = result['dgms']
    return diagrams

# =============================================================================
# 5. Subsampling (for computational efficiency)
# =============================================================================

def subsample_point_cloud(X, n_samples=1500, seed=42):
    """Subsample point cloud while preserving global geometry."""
    np.random.seed(seed)
    n = X.shape[0]
    if n <= n_samples:
        return X
    idx = np.random.choice(n, n_samples, replace=False)
    return X[np.sort(idx)]


def _plot_persistence_manual(diagrams, ax, title=''):
    """Manual persistence diagram plot (fallback when persim unavailable)."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # H0, H1, H2
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        dgm = np.array(dgm)
        finite = np.isfinite(dgm[:, 1])
        dgm_f = dgm[finite]
        dgm_inf = dgm[~finite]
        if len(dgm_f) > 0:
            ax.scatter(dgm_f[:, 0], dgm_f[:, 1], c=colors[dim % 3], label=f'$H_{dim}$', s=20)
        if len(dgm_inf) > 0:
            max_death = float(dgm_f[:, 1].max()) if len(dgm_f) > 0 else float(dgm_inf[:, 0].max()) * 1.2
            ax.scatter(dgm_inf[:, 0], [max_death] * len(dgm_inf), c=colors[dim % 3],
                      marker='o', s=20, edgecolors='k')
    ax_max = max(ax.get_xlim()[1], ax.get_ylim()[1]) if ax.get_xlim()[1] > 0 else 1
    ax.plot([0, ax_max], [0, ax_max], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    if title:
        ax.set_title(title)

# =============================================================================
# 6. Figure 1 Assembly
# =============================================================================

def create_figure1(save_path='figure1_tda_validation.png', dpi=150):
    """
    Create Figure 1: Topological Discrimination Between Ordered and Chaotic Dynamics.
    """
    # Simulation parameters
    dt = 0.02
    t_span = (0, 200)
    t_eval = np.arange(0, t_span[1], dt)
    
    # Discard transient: use last 60% of trajectory
    discard_frac = 0.4
    keep = int(len(t_eval) * (1 - discard_frac))
    t_eval = t_eval[-keep:]
    
    # Embedding parameters (as per manuscript)
    m = 3
    tau_vdp = 5   # delay for Van der Pol (samples)
    tau_lor = 8   # delay for Lorenz (samples)
    
    # --- Van der Pol ---
    y0_vdp = [1.0, 0.0]
    x_vdp = generate_time_series(van_der_pol, y0_vdp, t_span, t_eval, obs_idx=0)
    X_vdp = takens_embedding(x_vdp, m=m, tau=tau_vdp)
    X_vdp = subsample_point_cloud(X_vdp, n_samples=1500)
    
    diagrams_vdp = compute_persistence(X_vdp, maxdim=1)
    
    # --- Lorenz ---
    y0_lor = [1.0, 1.0, 1.0]
    x_lor = generate_time_series(lorenz, y0_lor, t_span, t_eval, obs_idx=0)
    X_lor = takens_embedding(x_lor, m=m, tau=tau_lor)
    X_lor = subsample_point_cloud(X_lor, n_samples=1500)
    
    diagrams_lor = compute_persistence(X_lor, maxdim=1)
    
    # --- Figure Layout ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Panel A: Van der Pol reconstructed phase space (3D -> 2D projection)
    ax_a = axes[0, 0]
    ax_a.scatter(X_vdp[:, 0], X_vdp[:, 1], c=X_vdp[:, 2], cmap='viridis', s=2, alpha=0.6)
    ax_a.set_xlabel('$x(t)$')
    ax_a.set_ylabel('$x(t+\\tau)$')
    ax_a.set_title('A. Van der Pol: Reconstructed Phase Space')
    ax_a.set_aspect('equal')
    ax_a.grid(True, alpha=0.3)
    
    # Panel B: Van der Pol persistence diagram
    ax_b = axes[0, 1]
    if HAS_PERSIM:
        plot_diagrams(diagrams_vdp, ax=ax_b, show=False)
    else:
        _plot_persistence_manual(diagrams_vdp, ax_b)
    ax_b.set_title('B. Van der Pol: Persistence Diagram')
    
    # Panel C: Lorenz reconstructed phase space
    ax_c = axes[1, 0]
    ax_c.scatter(X_lor[:, 0], X_lor[:, 1], c=X_lor[:, 2], cmap='plasma', s=2, alpha=0.6)
    ax_c.set_xlabel('$x(t)$')
    ax_c.set_ylabel('$x(t+\\tau)$')
    ax_c.set_title('C. Lorenz: Reconstructed Phase Space')
    ax_c.set_aspect('equal')
    ax_c.grid(True, alpha=0.3)
    
    # Panel D: Lorenz persistence diagram
    ax_d = axes[1, 1]
    if HAS_PERSIM:
        plot_diagrams(diagrams_lor, ax=ax_d, show=False)
    else:
        _plot_persistence_manual(diagrams_lor, ax_d)
    ax_d.set_title('D. Lorenz: Persistence Diagram')
    
    plt.suptitle(
        'Figure 1 | Validation of TDA sensitivity to dynamical regime.\n'
        'Time-delay embedding + persistent homology distinguishes ordered '
        'limit-cycle dynamics from deterministic chaos using only scalar time series.',
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved to {save_path}")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import os
    fig_dir = os.path.join(os.path.dirname(__file__), 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    create_figure1(save_path=os.path.join(fig_dir, 'fig1_tda_validation.png'), dpi=150)
