"""
RCDT 全脑动力学拓扑分析系统 V1.1
Author: Haolong Wang | 开发完成日期：2026年2月

优化: 参数集中管理，与稿中 Synthetic Parameter Table 一致，便于复现与敏感性分析。
论文对应初版程序为各脚本内散落的魔数；本模块为后续优化版本统一入口。
"""

import numpy as np

# =============================================================================
# Wilson–Cowan 与解剖/药理学（与稿中 Table 对应）
# =============================================================================
N_NODES = 30
TAU_E = 10.0       # ms, E 种群时间常数
TAU_I = 5.0        # ms, I 种群时间常数
W_EE = 1.2
W_IE = 1.0
W_EI = 1.0
W_II = 0.7
G_0 = 1.0
K_GAIN = 2.5
K_GAIN_RANGE = (0.5, 5.0)
SIGMA_NOISE = 0.02
V_CONDUCTION = 5.0  # mm/ms
DT = 0.005
TOTAL_TIME_MS = 60000
TRANSIENT_MS = 10000
D_CONCENTRATIONS = [0.0, 0.5, 1.0, 1.5, 2.0]

# =============================================================================
# TDA 管道默认值（初版 figure2 使用 tau=15, n_samples=1200）
# =============================================================================
EMBED_DIM = 3
TAU_EMBED_DEFAULT = 15       # 初版全脑仿真固定延迟（样本数）
N_SAMPLES_TDA_DEFAULT = 1200
N_SAMPLES_TDA_QUICK = 400
PE_H1_THRESHOLD = 0.01       # 分岔扫描中 H1 出现的 PE 阈值

# =============================================================================
# 优化: [D]_crit 定量定义并输出（稿中为定性描述）
# 定义方式: PE([D]) 曲线拐点 = 二阶差最大处，或首超 PE 阈值的 [D]
# =============================================================================
def compute_D_crit(D_vals, PE_vals, method='max_second_derivative'):
    """
    从 PE([D]) 曲线计算临界浓度 [D]_crit。

    Parameters
    ----------
    D_vals : array-like
        药物浓度序列
    PE_vals : array-like
        对应的 H1 持久熵
    method : str
        'max_second_derivative': [D]_crit = 二阶差最大处（拐点）
        'first_above_threshold': 首个 PE >= PE_threshold 的 [D]（需与 PE_H1_THRESHOLD 一致时用）

    Returns
    -------
    D_crit : float or None
        估计的 [D]_crit；无法估计时返回 None
    """
    D_vals = np.asarray(D_vals, dtype=float)
    PE_vals = np.asarray(PE_vals, dtype=float)
    if len(D_vals) < 3 or len(D_vals) != len(PE_vals):
        return None
    if method == 'max_second_derivative':
        # 拐点: 二阶差最大（离散二阶导）
        d2 = np.diff(PE_vals, 2)
        if len(d2) == 0:
            return None
        idx = np.argmax(d2)
        # 二阶差对应区间中点
        D_crit = float(0.5 * (D_vals[idx] + D_vals[idx + 2]))
        return D_crit
    if method == 'first_above_threshold':
        above = np.where(PE_vals >= PE_H1_THRESHOLD)[0]
        if len(above) == 0:
            return None
        return float(D_vals[above[0]])
    return None
