"""
Figure 2: Receptor-Weighted Whole-Brain Wilson-Cowan Model
RCDT Hypothesis - Pharmacologically Induced Topological Reorganization

Implements:
- 30-node Wilson-Cowan E-I dynamics with axonal delays
- Synthetic SC matrix (unit spectral radius) and 5-HT2A receptor map
- Gain modulation: G_i = G_0 + k * rho_i * [D]
- TDA pipeline (Takens + Ripser) on global mean E signal
- Receptor Shuffling control

Output: Figure 2 with brain graph + persistence diagrams across [D] levels
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import ripser

try:
    from persim import plot_diagrams
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

# =============================================================================
# Parameters (from RCDT manuscript)
# =============================================================================
N_NODES = 30
TAU_E = 10.0    # ms
TAU_I = 5.0     # ms
W_EE = 1.2
W_IE = 1.0
W_EI = 1.0
W_II = 0.7
G_0 = 1.0
K_GAIN = 2.5       # Increased from 1.0 to access bifurcation regime
K_GAIN_RANGE = (0.5, 5.0)  # For parameter sweep
SIGMA_NOISE = 0.02  # Brownian noise std (sqrt(dt) scaled)
V_CONDUCTION = 5.0  # m/s = mm/ms
DT = 0.005      # 5 ms per step (faster integration)
TOTAL_TIME_MS = 60000  # 60 s
TRANSIENT_MS = 10000   # discard first 10 s
D_CONCENTRATIONS = [0.0, 0.5, 1.0, 1.5, 2.0]

# =============================================================================
# 1. Synthetic Data Generation
# =============================================================================

def create_synthetic_sc(n_nodes=30, seed=42):
    """
    Create synthetic 30x30 Structural Connectivity matrix.
    Symmetric, sparse-ish, normalized to unit spectral radius.
    """
    np.random.seed(seed)
    # Random connectivity with distance-dependent decay
    pos = np.random.randn(n_nodes, 3) * 40  # node positions in mm
    D = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    D = np.maximum(D, 20)  # min distance 20 mm
    # Inverse distance weighting (closer = stronger)
    C = np.exp(-D / 80) * (np.random.rand(n_nodes, n_nodes) > 0.6)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    # Normalize to unit spectral radius
    rho = np.max(np.abs(np.linalg.eigvals(C)))
    C = C / (rho + 1e-10)
    return C, D

def create_synthetic_receptor_map(n_nodes=30, seed=42):
    """
    Synthetic 5-HT2A receptor density mimicking Beliveau (2017).
    Higher in DMN-like nodes (e.g., posterior-medial, mPFC indices).
    """
    np.random.seed(seed)
    # DMN-like regions: indices 0-4 (medial), 8-12 (posterior), 20-24 (lateral)
    rho = np.ones(n_nodes) * 0.3
    dmn_indices = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 20, 21, 22]
    rho[dmn_indices] = np.linspace(0.7, 1.0, len(dmn_indices))
    rho += np.random.rand(n_nodes) * 0.15
    rho = np.maximum(rho, 0.1)
    rho = rho / rho.max()  # normalize to [0, 1]
    return rho

def compute_delay_matrix(D, v=V_CONDUCTION):
    """τ_ij = D_ij / v (in ms)."""
    return D / v

# =============================================================================
# 2. Wilson-Cowan Integration with Delays
# =============================================================================

def sigmoid(x):
    """S(x) = 1 / (1 + exp(-x)). Clipped for stability."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def run_wilson_cowan(C, D, rho, D_conc, G_0=G_0, k=K_GAIN,
                     tau_E=TAU_E, tau_I=TAU_I, dt=DT, t_total=TOTAL_TIME_MS,
                     transient_ms=TRANSIENT_MS, sigma_noise=SIGMA_NOISE, seed=42):
    """
    Integrate Wilson-Cowan equations with axonal delays using Euler-Maruyama.
    Adds Brownian noise: dE += sigma*sqrt(dt)*dW to escape stable fixed points.

    dE_i/dt = (-E_i + S(...)) / tau_E + sigma * dW_E
    dI_i/dt = (-I_i + S(...)) / tau_I + sigma * dW_I

    G_i = G_0 + k * rho_i * [D]
    """
    np.random.seed(seed)
    n = C.shape[0]
    tau_ij = compute_delay_matrix(D)
    delay_steps_int = np.round(tau_ij / dt).astype(int)
    max_delay = min(int(50 / dt), 100)  # cap at 100 steps
    delay_steps_int = np.clip(delay_steps_int, 0, max_delay)

    G = G_0 + k * rho * D_conc
    E = 0.1 + 0.3 * np.random.rand(n)
    I = 0.05 + 0.15 * np.random.rand(n)

    n_steps = int(t_total / dt)
    discard_steps = int(transient_ms / dt)
    n_keep = n_steps - discard_steps
    buf_size = max_delay + 1
    # Downsample output: keep every nth point to reduce memory (we need ~50k pts for TDA)
    downsample = max(1, n_keep // 50000)
    n_out = (n_keep + downsample - 1) // downsample
    E_out = np.zeros((n_out, n))

    E_buf = np.zeros((buf_size, n))
    E_buf[0] = E.copy()

    P = 0.5

    for step in range(1, n_steps + 1):
        steps_back = np.clip(delay_steps_int, 0, min(step, max_delay))
        # E at step (step - d) lives in buffer slot (step - d) % buf_size
        buf_idx = np.where(step - steps_back >= 0, (step - steps_back) % buf_size, 0)
        idx_rows = buf_idx.ravel()
        idx_cols = np.repeat(np.arange(n), n)
        E_delayed = E_buf[idx_rows, idx_cols].reshape(n, n)

        input_E = W_EE * E - W_IE * I + (C * E_delayed).sum(axis=1) + P
        input_E = G * input_E

        dE = (-E + sigmoid(input_E)) / tau_E
        dI = (-I + sigmoid(W_EI * E - W_II * I)) / tau_I

        # Euler-Maruyama: add Brownian noise (sqrt(dt) for proper scaling)
        if sigma_noise > 0:
            dW = np.random.randn(2 * n).astype(np.float32)
            dW_E, dW_I = dW[:n], dW[n:]
            E = E + dt * dE + sigma_noise * np.sqrt(dt) * dW_E
            I = I + dt * dI + sigma_noise * np.sqrt(dt) * dW_I
        else:
            E = E + dt * dE
            I = I + dt * dI
        E = np.clip(E, 0, 1)
        I = np.clip(I, 0, 1)
        E_buf[step % buf_size] = E.copy()

        if step > discard_steps:
            idx = step - discard_steps - 1
            if idx % downsample == 0:
                E_out[idx // downsample] = E.copy()

    return E_out

# =============================================================================
# 3. Global Observable & TDA Pipeline
# =============================================================================

def global_mean_E(E_trajectory):
    """Extract global mean excitatory activity (scalar time series)."""
    return E_trajectory.mean(axis=1)

def takens_embedding(x, m=3, tau=10):
    """Time-delay embedding."""
    n = len(x)
    N_embed = n - (m - 1) * tau
    if N_embed <= 0:
        raise ValueError("Time series too short for embedding")
    X = np.zeros((N_embed, m))
    for i in range(m):
        X[:, i] = x[i * tau : i * tau + N_embed]
    return X

def compute_persistence(point_cloud, maxdim=1):
    """Vietoris-Rips persistence via Ripser."""
    result = ripser.ripser(point_cloud, maxdim=maxdim)
    return result['dgms']

def persistent_entropy(diagrams, dim=1):
    """
    Compute Persistent Entropy from persistence diagram (H_dim).
    PE = -sum(p_i * log(p_i)) where p_i = l_i/L, l_i = death - birth (lifetime).
    Uses natural log. Higher PE = more dispersed lifetimes (fragmented topology).
    """
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0.0
    dgm = np.array(diagrams[dim])
    finite = np.isfinite(dgm[:, 1])
    if not np.any(finite):
        return 0.0
    lifetimes = (dgm[finite, 1] - dgm[finite, 0]).astype(float)
    lifetimes = lifetimes[lifetimes > 1e-12]  # exclude numerical zeros
    if len(lifetimes) == 0:
        return 0.0
    L = lifetimes.sum()
    if L <= 0:
        return 0.0
    p = lifetimes / L
    return float(-np.sum(p * np.log(p + 1e-15)))

def subsample_point_cloud(X, n_samples=1200, seed=42):
    """Subsample for computational efficiency."""
    np.random.seed(seed)
    n = X.shape[0]
    if n <= n_samples:
        return X
    idx = np.random.choice(n, n_samples, replace=False)
    return X[np.sort(idx)]

# =============================================================================
# 4. Receptor Shuffling
# =============================================================================

def shuffle_receptor_map(rho, seed=None):
    """Permute rho to create control condition."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rho))
    return rho[perm], perm

# =============================================================================
# 5. Figure 2 Assembly
# =============================================================================

def _create_figure3_persistent_entropy(results_exp, results_shuffled, save_path):
    """
    Figure 3: Persistent Entropy vs drug concentration [D].
    Non-linear jump supports phase transition hypothesis.
    """
    out_dir = os.path.dirname(save_path)
    pe_path = os.path.join(out_dir, 'fig3_persistent_entropy.png') if out_dir else 'fig3_persistent_entropy.png'

    fig, ax = plt.subplots(figsize=(6, 4))
    D_vals = [r['D'] for r in results_exp]
    PE_exp = [r['PE'] for r in results_exp]
    ax.plot(D_vals, PE_exp, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Experimental ($\\rho$ true)')

    if results_shuffled:
        PE_shuf = [r['PE'] for r in results_shuffled]
        ax.plot(D_vals, PE_shuf, 's--', color='#ff7f0e', linewidth=2, markersize=8, label='Shuffled ($\\rho_{\\pi}$)')

    ax.set_xlabel('Drug concentration $[D]$', fontsize=12)
    ax.set_ylabel('Persistent Entropy (H₁)', fontsize=12)
    ax.set_title('Figure 3 | Persistent Entropy quantifies topological phase transition', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(D_vals) - 0.1, max(D_vals) + 0.1)
    plt.tight_layout()
    plt.savefig(pe_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 (Persistent Entropy) saved to {pe_path}")

def _plot_persistence_manual(diagrams, ax):
    """Manual persistence plot (fallback)."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        dgm = np.array(dgm)
        finite = np.isfinite(dgm[:, 1])
        dgm_f = dgm[finite]
        dgm_inf = dgm[~finite]
        if len(dgm_f) > 0:
            ax.scatter(dgm_f[:, 0], dgm_f[:, 1], c=colors[dim % 3], s=15, alpha=0.8)
        if len(dgm_inf) > 0:
            max_d = float(dgm_f[:, 1].max()) if len(dgm_f) > 0 else float(dgm_inf[:, 0].max()) * 1.2
            ax.scatter(dgm_inf[:, 0], [max_d] * len(dgm_inf), c=colors[dim % 3],
                      marker='o', s=15, edgecolors='k')
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1], 0.01)]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')

def create_figure2(save_path='fig2_receptor_topology.png', dpi=150,
                  run_shuffled=False, shuffled_seed=123, quick_mode=False):
    """
    Generate Figure 2: Receptor-weighted pharmacological perturbation induces
    global topological reorganization.
    """
    # Create synthetic data
    C, D = create_synthetic_sc(N_NODES)
    rho = create_synthetic_receptor_map(N_NODES)
    pos_2d = _get_node_positions_2d(D)  # for brain graph layout

    n_samples_tda = 400 if quick_mode else 1200
    results_exp = []
    for d_conc in D_CONCENTRATIONS:
        E_traj = run_wilson_cowan(C, D, rho, d_conc, t_total=TOTAL_TIME_MS, transient_ms=TRANSIENT_MS)
        x_global = global_mean_E(E_traj)
        X_emb = takens_embedding(x_global, m=3, tau=15)
        X_sub = subsample_point_cloud(X_emb, n_samples=n_samples_tda)
        dgms = compute_persistence(X_sub, maxdim=1)
        pe = persistent_entropy(dgms, dim=1)
        results_exp.append({'D': d_conc, 'x': x_global, 'dgms': dgms, 'X': X_sub, 'PE': pe})

    # Optional: Receptor Shuffling control
    results_shuffled = None
    if run_shuffled:
        rho_shuf, _ = shuffle_receptor_map(rho, shuffled_seed)
        results_shuffled = []
        for d_conc in D_CONCENTRATIONS:
            E_traj = run_wilson_cowan(C, D, rho_shuf, d_conc, t_total=TOTAL_TIME_MS, transient_ms=TRANSIENT_MS)
            x_global = global_mean_E(E_traj)
            X_emb = takens_embedding(x_global, m=3, tau=15)
            X_sub = subsample_point_cloud(X_emb, n_samples=n_samples_tda)
            dgms = compute_persistence(X_sub, maxdim=1)
            pe = persistent_entropy(dgms, dim=1)
            results_shuffled.append({'D': d_conc, 'dgms': dgms, 'PE': pe})

    # Figure 3: Persistent Entropy vs [D] (quantifies phase transition)
    _create_figure3_persistent_entropy(results_exp, results_shuffled, save_path)

    # Figure layout: Panel A (brain graph) + Panels B-F (persistence diagrams)
    n_panels = 1 + len(D_CONCENTRATIONS)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Brain graph colored by receptor density
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_brain_graph(ax_a, C, pos_2d, rho)
    ax_a.set_title('A. Structural Connectivity\n(node color = $\\rho_i$ receptor density)')

    # Panels B-F: Persistence diagrams for each [D]
    panel_labels = 'BCDEFGH'[:len(results_exp)]
    for idx, res in enumerate(results_exp):
        row, col = (idx + 1) // 3, (idx + 1) % 3
        ax = fig.add_subplot(gs[row, col])
        if HAS_PERSIM:
            plot_diagrams(res['dgms'], ax=ax, show=False)
        else:
            _plot_persistence_manual(res['dgms'], ax)
        ax.set_title(f'{panel_labels[idx]}. [D] = {res["D"]}')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')

    plt.suptitle(
        'Figure 2 | Receptor-weighted pharmacological perturbation induces global topological reorganization.\n'
        'Whole-brain Wilson–Cowan model with heterogeneous gain modulation. '
        'Topological transitions detected via Takens + persistent homology.',
        fontsize=10, y=1.02
    )
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {save_path}")

    # Optional: Save shuffled comparison figure
    if run_shuffled:
        fig2 = plt.figure(figsize=(12, 5))
        for idx, (res_exp, res_shuf) in enumerate(zip(results_exp, results_shuffled)):
            ax1 = fig2.add_subplot(2, 5, idx + 1)
            ax2 = fig2.add_subplot(2, 5, idx + 6)
            if HAS_PERSIM:
                plot_diagrams(res_exp['dgms'], ax=ax1, show=False)
                plot_diagrams(res_shuf['dgms'], ax=ax2, show=False)
            else:
                _plot_persistence_manual(res_exp['dgms'], ax1)
                _plot_persistence_manual(res_shuf['dgms'], ax2)
            ax1.set_title(f'Experimental [D]={res_exp["D"]}')
            ax2.set_title(f'Shuffled [D]={res_exp["D"]}')
        plt.suptitle('Receptor Shuffling Control: Experimental vs Shuffled $\\rho$', fontsize=11)
        plt.tight_layout()
        out_dir = os.path.dirname(save_path)
        shuffle_path = os.path.join(out_dir, 'fig2_supp_shuffled_control.png') if out_dir else 'fig2_supp_shuffled_control.png'
        plt.savefig(shuffle_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Shuffled control figure saved to {shuffle_path}")

    return results_exp, results_shuffled

def _get_node_positions_2d(D):
    """2D layout from distance matrix (MDS-like)."""
    n = D.shape[0]
    # Use first 2 eigenvectors of -0.5 * J @ D^2 @ J (classic MDS)
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    evals, evecs = eigh(B)
    idx = np.argsort(evals)[::-1][:2]
    pos = evecs[:, idx] * np.sqrt(np.maximum(evals[idx], 0))
    return pos

def _plot_brain_graph(ax, C, pos, rho):
    """Plot connectivity graph with nodes colored by rho."""
    n = C.shape[0]
    # Draw edges (thickness by weight)
    threshold = np.percentile(C[C > 0], 50) if (C > 0).any() else 0
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > threshold:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        'k-', alpha=0.3 * C[i, j], linewidth=0.5)
    # Draw nodes
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=rho, s=80, cmap='YlOrRd', vmin=0, vmax=1, edgecolors='black')
    plt.colorbar(sc, ax=ax, label='$\\rho_i$ (5-HT2A)')
    ax.set_aspect('equal')
    ax.axis('off')

# =============================================================================
# 6. Bifurcation Parameter Sweep
# =============================================================================

def run_bifurcation_sweep(n_k=8, D_fixed=1.0, sigma_noise=SIGMA_NOISE,
                         t_total=6000, transient=2000, n_samples_tda=400,
                         pe_threshold=0.01, seed=42):
    """
    Parameter sweep to find k threshold where H1 features (loops) first emerge.
    Returns (k_values, PE_values, k_crit_estimate).
    """
    np.random.seed(seed)
    C, D = create_synthetic_sc(N_NODES, seed=seed)
    rho = create_synthetic_receptor_map(N_NODES, seed=seed)

    k_min, k_max = K_GAIN_RANGE
    k_values = np.linspace(k_min, k_max, n_k)
    PE_values = []

    print("Bifurcation sweep: scanning k from", k_min, "to", k_max)
    for i, k in enumerate(k_values):
        E_traj = run_wilson_cowan(C, D, rho, D_fixed, k=k, sigma_noise=sigma_noise,
                                  t_total=t_total, transient_ms=transient, seed=seed + i)
        x = global_mean_E(E_traj)
        X = takens_embedding(x, m=3, tau=12)
        X_sub = subsample_point_cloud(X, n_samples=n_samples_tda, seed=seed + i)
        dgms = compute_persistence(X_sub, maxdim=1)
        pe = persistent_entropy(dgms, dim=1)
        PE_values.append(pe)
        print(f"  k={k:.2f} -> PE(H1)={pe:.4f}")

    PE_values = np.array(PE_values)
    # First k where PE exceeds threshold
    above = np.where(PE_values >= pe_threshold)[0]
    k_crit = float(k_values[above[0]]) if len(above) > 0 else None

    return k_values, PE_values, k_crit

def plot_bifurcation_sweep(k_values, PE_values, k_crit, save_path='figure_bifurcation_sweep.png'):
    """Plot k vs PE and mark estimated bifurcation threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, PE_values, 'o-', color='#1f77b4', linewidth=2, markersize=6)
    if k_crit is not None:
        ax.axvline(k_crit, color='red', linestyle='--', alpha=0.8, label=f'$k_{{crit}} \\approx {k_crit:.2f}$')
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coupling sensitivity $k$', fontsize=12)
    ax.set_ylabel('Persistent Entropy (H₁)', fontsize=12)
    ax.set_title('Bifurcation sweep: H₁ emergence threshold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bifurcation sweep plot saved to {save_path}")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffled', action='store_true', help='Run receptor shuffling control')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _fig_dir = os.path.join(_script_dir, 'figs')
    os.makedirs(_fig_dir, exist_ok=True)
    parser.add_argument('--output', default=os.path.join(_fig_dir, 'fig2_receptor_topology.png'))
    parser.add_argument('--quick', action='store_true', help='Shorten simulation for testing')
    parser.add_argument('--sweep', action='store_true', help='Run bifurcation parameter sweep (k vs PE)')
    args = parser.parse_args()

    if args.sweep:
        k_vals, pe_vals, k_crit = run_bifurcation_sweep()
        out_dir = os.path.dirname(args.output)
        sweep_path = os.path.join(out_dir, 'fig2_supp_bifurcation_sweep.png') if out_dir else 'fig2_supp_bifurcation_sweep.png'
        plot_bifurcation_sweep(k_vals, pe_vals, k_crit, save_path=sweep_path)
        print(f"Estimated bifurcation threshold: k_crit = {k_crit}")
        exit(0)

    if args.quick:
        globals()['TOTAL_TIME_MS'] = 5000   # 5 s
        globals()['TRANSIENT_MS'] = 2000
        globals()['D_CONCENTRATIONS'] = [0.0, 1.0, 2.0]
    create_figure2(save_path=args.output, run_shuffled=args.shuffled, quick_mode=args.quick)
