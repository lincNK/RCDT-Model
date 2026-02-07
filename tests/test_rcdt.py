"""
RCDT 核心函数单元测试。优化: 保证重构后 TDA/参数/动力学行为可回归。
运行: python -m unittest tests.test_rcdt  或  python tests/test_rcdt.py
"""

import sys
import os
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rcdt_params import compute_D_crit
from rcdt_tda import (
    takens_embedding,
    persistent_entropy,
    subsample_point_cloud,
    surrogate_phase_randomize,
    tau_first_min_autocorr,
)


class TestRCDT(unittest.TestCase):
    def test_takens_embedding_shape(self):
        x = np.random.randn(1000)
        X = takens_embedding(x, m=3, tau=5)
        self.assertEqual(X.shape[0], 1000 - (3 - 1) * 5)
        self.assertEqual(X.shape[1], 3)

    def test_persistent_entropy_known(self):
        diagrams = [np.array([]), np.array([[0.0, 1.0]])]
        pe = persistent_entropy(diagrams, dim=1)
        self.assertAlmostEqual(pe, 0.0, places=10)
        diagrams = [np.array([]), np.array([[0.0, 1.0], [0.0, 1.0]])]
        pe = persistent_entropy(diagrams, dim=1)
        self.assertAlmostEqual(pe, np.log(2), places=4)

    def test_subsample_point_cloud(self):
        X = np.random.randn(2000, 3)
        Y = subsample_point_cloud(X, n_samples=500, seed=42)
        self.assertEqual(Y.shape[0], 500)
        self.assertEqual(Y.shape[1], 3)
        Y2 = subsample_point_cloud(X, n_samples=500, seed=42)
        np.testing.assert_array_almost_equal(Y, Y2)

    def test_surrogate_phase_randomize(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        y = surrogate_phase_randomize(x, seed=42)
        self.assertEqual(y.shape, x.shape)

    def test_tau_first_min_autocorr(self):
        x = np.sin(np.linspace(0, 50 * np.pi, 1000))
        tau = tau_first_min_autocorr(x, tau_max=80)
        self.assertGreaterEqual(tau, 1)
        self.assertLessEqual(tau, 80)

    def test_compute_D_crit(self):
        D = [0.0, 0.5, 1.0, 1.5, 2.0]
        PE = [0.001, 0.002, 0.015, 0.04, 0.05]
        d_crit = compute_D_crit(D, PE, method='max_second_derivative')
        self.assertIsNotNone(d_crit)
        self.assertTrue(0.5 <= d_crit <= 1.5)

    def test_wilson_cowan_output_shape(self):
        from figure2_simulation import create_synthetic_sc, create_synthetic_receptor_map, run_wilson_cowan
        C, D = create_synthetic_sc(30, seed=42)
        rho = create_synthetic_receptor_map(30, seed=42)
        E_out = run_wilson_cowan(C, D, rho, 0.5, t_total=100.0, transient_ms=20.0, seed=42)
        self.assertEqual(E_out.shape[1], 30)
        self.assertEqual(E_out.ndim, 2)


if __name__ == "__main__":
    unittest.main()
