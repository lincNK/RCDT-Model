"""
RCDT 全脑动力学拓扑分析系统 V1.1
Author: Haolong Wang | 开发完成日期：2026年2月

优化: 将 TDA 管道抽离为公共模块，供 figure1 与 figure2 复用，保证嵌入/持久同调/持久熵
计算一致；初版程序在两脚本中各自实现，易产生 τ/m 等参数不一致。
"""

import numpy as np

try:
    import ripser
except ImportError:
    ripser = None


def takens_embedding(x, m=3, tau=1):
    """
    标量时间序列的 Takens 时间延迟嵌入。
    X(t) = [x(t), x(t+τ), ..., x(t+(m-1)τ)]
    """
    x = np.asarray(x).ravel()
    n = len(x)
    N_embed = n - (m - 1) * tau
    if N_embed <= 0:
        raise ValueError("Time series too short for embedding (m=%d, tau=%d, n=%d)" % (m, tau, n))
    X = np.zeros((N_embed, m))
    for i in range(m):
        X[:, i] = x[i * tau : i * tau + N_embed]
    return X


def tau_first_min_autocorr(x, tau_max=50):
    """
    优化: 嵌入延迟 τ 由自相关函数第一个局部最小选取（稿中建议 mutual information
    或 first minimum of autocorrelation）；初版为各图 ad hoc 固定值。
    Returns tau in sample index (int).
    """
    x = np.asarray(x).ravel()
    n = len(x)
    x = x - np.mean(x)
    c0 = np.dot(x, x)
    if c0 <= 0:
        return 1
    acf = np.array([np.dot(x[:n - k], x[k:]) / c0 for k in range(min(tau_max + 1, n // 2))])
    # 第一个局部最小：acf[i] < acf[i-1] and acf[i] < acf[i+1]
    for i in range(1, len(acf) - 1):
        if acf[i] <= acf[i - 1] and acf[i] <= acf[i + 1]:
            return max(1, i)
    return max(1, np.argmin(acf[1:]) + 1)


def compute_persistence(point_cloud, maxdim=1):
    """Vietoris–Rips 持久同调（Ripser）。返回 list of (birth, death) arrays for H0, H1, ..."""
    if ripser is None:
        raise ImportError("ripser is required for compute_persistence")
    result = ripser.ripser(point_cloud, maxdim=maxdim)
    return result['dgms']


def persistent_entropy(diagrams, dim=1, treat_infinite_lifetime=None):
    """
    H_dim 持久熵 PE = -Σ p_i ln(p_i)，p_i = l_i/L，l_i = death - birth。
    优化: 无穷长条（death=inf）可选用 treat_infinite_lifetime 截断或排除，与文献一致；
    初版仅用有限 lifetime，未显式说明 inf 处理。
    """
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0.0
    dgm = np.array(diagrams[dim])
    finite = np.isfinite(dgm[:, 1])
    lifetimes = (dgm[finite, 1] - dgm[finite, 0]).astype(float)
    # 无穷长条：可选截断为最大有限 death 的倍数
    if not np.all(finite) and treat_infinite_lifetime is not None:
        max_finite = np.max(dgm[finite, 1]) if np.any(finite) else 1.0
        for i in range(len(dgm)):
            if not finite[i]:
                lifetimes = np.append(lifetimes, treat_infinite_lifetime * max_finite)
    lifetimes = lifetimes[lifetimes > 1e-12]
    if len(lifetimes) == 0:
        return 0.0
    L = lifetimes.sum()
    if L <= 0:
        return 0.0
    p = lifetimes / L
    return float(-np.sum(p * np.log(p + 1e-15)))


def subsample_point_cloud(X, n_samples=1200, seed=42):
    """点云随机子采样以控制计算量，seed 可复现。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= n_samples:
        return X
    idx = rng.choice(n, n_samples, replace=False)
    return X[np.sort(idx)]


def surrogate_phase_randomize(x, seed=None):
    """
    优化: 替代数据（相位随机化），用于检验拓扑是否区别于纯噪声（稿中可证伪性）。
    保持功率谱，破坏相位关系。返回与 x 等长的 1D 数组。
    """
    x = np.asarray(x).ravel()
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 4:
        return x.copy()
    X = np.fft.rfft(x)
    phase = np.angle(X)
    # 保持 DC 和 Nyquist 相位，其余随机化
    new_phase = phase.copy()
    idx = np.arange(1, len(phase) - (1 if n % 2 == 0 else 0), dtype=int)
    new_phase[idx] = rng.uniform(-np.pi, np.pi, size=len(idx))
    Y = np.abs(X) * np.exp(1j * new_phase)
    return np.fft.irfft(Y, n=n).real
