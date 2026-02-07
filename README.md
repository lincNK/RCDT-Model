# Receptor-Constrained Dynamical Topology (RCDT) Hypothesis

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.04.703742-B31B1B)](https://doi.org/10.64898/2026.02.04.703742)  
**Code version:** V1.1 (optimised release; preprint figures reproducible with default `--seed 42`). See [CHANGELOG.md](CHANGELOG.md).

**Preprint now live on bioRxiv.** A computational framework bridging molecular pharmacology and whole-brain dynamics through topological data analysis. The RCDT hypothesis proposes that receptor-weighted gain modulation induces **topological phase transitions** in neural attractors—detectable as non-linear jumps in Persistent Entropy across drug concentration.

---

## Central Result: Topological Phase Transition

The key prediction of the RCDT hypothesis is a **non-linear jump** in Persistent Entropy (H₁) as drug concentration [D] crosses a critical threshold. Below [D]<sub>crit</sub>, the attractor remains low-dimensional; above it, the topology fragments—consistent with *ego dissolution* as operational Betti-1 stability breakdown.

<p align="center"><img src="figs/fig3_persistent_entropy.png" alt="Persistent Entropy vs drug concentration" width="600"></p>

*Figure 3 | Persistent Entropy vs drug concentration [D]. The experimental curve (true receptor map ρ) is compared against the shuffled control (ρ<sub>π</sub>). A non-linear jump supports the phase transition hypothesis; divergence between curves supports 5-HT2A receptor-specificity.*

---

## Overview

- **Model**: 30-node Wilson–Cowan E-I dynamics on structural connectivity with axonal delays
- **Pharmacology**: Gain modulation G<sub>i</sub> = G<sub>0</sub> + k·ρ<sub>i</sub>·[D] weighted by 5-HT2A receptor density
- **Topology**: Takens embedding + Vietoris–Rips persistent homology
- **Integration**: Euler–Maruyama with small Brownian noise (prevents fixed-point trapping)

See [`RCDT_bioRxiv_manuscript.md`](RCDT_bioRxiv_manuscript.md) for the full manuscript.

---

## Installation

```bash
git clone https://github.com/lincNK/RCDT-Model.git
cd RCDT-Model
pip install -r requirements.txt
```

**Requirements**: `numpy`, `scipy`, `matplotlib`, `ripser`, `persim`

---

## Usage

### 统一入口（推荐）

```bash
python main.py figure1              # Figure 1：TDA 校验
python main.py figure2              # Figure 2/3：全脑仿真与持久熵
python main.py figure2 --quick      # 快速模式
python main.py figure2 --shuffled   # 含受体洗牌对照
python main.py figure2 --sweep      # 分岔参数扫描
python main.py figure2 --surrogate  # 替代数据检验（相位随机化）
python main.py figure2 --seed 42    # 固定种子复现（默认 42）
```

**复现论文图**：使用默认 `--seed 42` 即可复现与预印本一致的输出；Figure 3 会额外打印定量 **[D]_crit**（拐点估计）。

### Figure 1: TDA Pipeline Calibration (Van der Pol vs Lorenz)

```bash
python figure1_persistence_diagram.py
```

Output: `figs/fig1_tda_validation.png`

### Figure 2 & 3: Whole-Brain Model and Persistent Entropy

```bash
# Main simulation (5 concentration levels, outputs to figs/)
python figure2_simulation.py

# Quick mode (3 levels, shorter simulation)
python figure2_simulation.py --quick

# Include receptor shuffling control
python figure2_simulation.py --shuffled

# Bifurcation parameter sweep (find k_crit)
python figure2_simulation.py --sweep
```

Outputs (all saved to `figs/`):
- `figs/fig2_receptor_topology.png` — Brain graph + persistence diagrams
- `figs/fig3_persistent_entropy.png` — PE vs [D] curve
- `figs/fig2_supp_shuffled_control.png` — Experimental vs shuffled (with `--shuffled`)
- `figs/fig2_supp_bifurcation_sweep.png` — k vs PE sweep (with `--sweep`)

---

## Figure Outputs

| File | Description |
|------|-------------|
| `figs/fig1_tda_validation.png` | TDA calibration: limit cycle vs chaos discrimination |
| `figs/fig2_receptor_topology.png` | Receptor-weighted topology across [D] |
| `figs/fig3_persistent_entropy.png` | **Persistent Entropy vs [D]** (phase transition curve) |
| `figs/fig2_supp_shuffled_control.png` | Receptor shuffling control |
| `figs/fig2_supp_bifurcation_sweep.png` | Bifurcation threshold sweep |

---

## 优化说明（相对论文初版程序）

- **参数集中**：`rcdt_params.py` 统一 WC/TDA 参数，与稿中 Parameter Table 对应；**[D]_crit** 由 PE([D]) 拐点定量计算并输出。
- **公共 TDA 模块**：`rcdt_tda.py` 提供 Takens 嵌入、持久同调、持久熵、子采样；支持 **τ 自相关第一最小** 选取与 **相位随机化替代数据**。
- **可复现**：入口与脚本支持 `--seed`（默认 42），复现论文图。
- **扩展分析**：`figure2` 可选 `--surrogate`、`--n-shuffles`、`--n-seeds-sweep`，用于替代数据检验与洗牌/k_crit 稳定性。

## Testing

```bash
python -m unittest tests.test_rcdt -v
```

## Contributing

Constructive feedback and collaborations are welcome. Please open an **issue** for theoretical discussions or **pull requests** for model optimizations.

---

## Citation

If you use this code or the RCDT hypothesis in your research, please cite our preprint:

```bibtex
@article{Wang2026RCDT,
  title={Topological Phase Transitions in Whole-Brain Dynamics Driven by Spatially Heterogeneous Receptor Gain Modulation: A Receptor-Constrained Dynamical Topology (RCDT) Hypothesis},
  author={Haolong Wang},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.64898/2026.02.04.703742},
  url={https://doi.org/10.64898/2026.02.04.703742}
}
```

---

## Releases

- **v1.0**: Code as in the bioRxiv preprint.
- **v1.1**: Current optimised version (centralised params/TDA, reproducibility, optional surrogate & multi-shuffle; see [CHANGELOG.md](CHANGELOG.md)).

Tags: `v1.0`, `v1.1` on the [Releases](https://github.com/lincNK/RCDT-Model/releases) page.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
