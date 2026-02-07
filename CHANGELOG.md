# Changelog

All notable changes to the RCDT codebase are listed here. The project follows a simple version scheme: **V1.x** for backward-compatible updates; paper (bioRxiv) corresponds to the initial **V1.0** behaviour under default options.

---

## [V1.1] — 2026-02

**Labelled as an optimized follow-up to the preprint release (V1.0).** Scientific results and main figures are unchanged when using default `--seed 42`; new options add optional analyses only.

### Added

- **`rcdt_params.py`**: Centralised parameters (Wilson–Cowan, TDA defaults) aligned with the manuscript Parameter Table; **`compute_D_crit()`** for quantitative critical concentration from the PE([D]) curve (inflection point).
- **`rcdt_tda.py`**: Shared TDA pipeline (Takens, persistence, persistent entropy, subsampling); **`tau_first_min_autocorr()`** for delay selection; **`surrogate_phase_randomize()`** for surrogate data checks.
- **Reproducibility**: `--seed` (default 42) on `main.py` and scripts; consistent use of `np.random.default_rng(seed)` in figure2.
- **Extended analyses** (figure2): `--surrogate` (phase-randomised control), `--n-shuffles` (multiple shuffle replicates), `--n-seeds-sweep` (k_crit stability); Figure 3 prints **[D]_crit** and can plot surrogate / shuffle bands.
- **Tests**: `tests/test_rcdt.py` for core TDA, params, and Wilson–Cowan output shape.
- **README**: Reproducibility note, optimisation summary, and testing section.

### Changed

- **figure1 / figure2**: Use `rcdt_params` and `rcdt_tda` instead of local constants and duplicate TDA code; in-code comments mark “优化” (optimisation) vs “初版” (original) where relevant.
- **main.py**: Forwards `--seed`, `--surrogate`, `--n-shuffles`, `--n-seeds-sweep` to figure2.

### Unchanged (vs V1.0)

- Default pipeline (embedding delay τ, concentrations, integration, noise) and main figures (Fig 1–3) match the preprint when run with default seed.
- No new dependencies; same `requirements.txt` scope.

---

## [V1.0] — 2026-02

- Initial release corresponding to the bioRxiv preprint (2026.02.04.703742).
- Figure 1: TDA calibration (Van der Pol / Lorenz).
- Figure 2/3: Whole-brain Wilson–Cowan model, receptor shuffling control, bifurcation sweep, persistent entropy vs [D].
- Standalone scripts plus `main.py` launcher.
