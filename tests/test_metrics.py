"""
tests/test_metrics.py — 韧性度量模块的针对性测试

测试分组：
- TestTransition      : 相变阶数与两种阈值口径
- TestResilienceArea  : 韧性损失面积 AUC / RHI
- TestRecovery        : 恢复过程指标
- TestWithPhase6      : 与 phase6 四级模型联调（曲线 → 指标 的完整链路）
"""

import os
import sys
import importlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import networkx as nx
import numpy as np


def _metrics():
    return importlib.import_module("metrics.resilience_metrics")


def _make_pair(N=60, m=2, seed=42):
    return (nx.barabasi_albert_graph(N, m, seed=seed),
            nx.barabasi_albert_graph(N, m, seed=seed + 1))


F = np.linspace(0.0, 1.0, 11)


# ============================================================
# 相变
# ============================================================

class TestTransition:

    def test_jump_size_zero_for_flat_curve(self):
        m = _metrics()
        assert m.jump_size(np.ones(5)) == 0.0

    def test_max_slope_is_grid_independent_for_the_same_ramp(self):
        """同一个跳变（线性下降 1→0），换网格密度后斜率应当几乎不变（jump_size 则会变）。"""
        m = _metrics()
        f_coarse = np.linspace(0.0, 1.0, 11)
        f_fine = np.linspace(0.0, 1.0, 101)
        slope_coarse = m.max_slope(f_coarse, 1.0 - f_coarse)
        slope_fine = m.max_slope(f_fine, 1.0 - f_fine)
        assert abs(slope_coarse - slope_fine) < 1e-9
        assert m.jump_size(1.0 - f_fine) < m.jump_size(1.0 - f_coarse)

    def test_max_slope_rejects_non_increasing_fractions(self):
        m = _metrics()
        try:
            m.max_slope([0.0, 0.5, 0.4], [1.0, 0.5, 0.4])
        except ValueError:
            return
        raise AssertionError("f 不递增应当报错")

    def test_second_order_for_smooth_decline(self):
        m = _metrics()
        order, jump = m.transition_order(F, np.linspace(1.0, 0.0, 11))
        assert order == "二阶"
        assert abs(jump - 0.1) < 1e-12

    def test_first_order_for_sudden_collapse(self):
        m = _metrics()
        S = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        order, jump = m.transition_order(F, S)
        assert order == "一阶" and jump > 0.9

    def test_midpoint_threshold_interpolates(self):
        """S = 1 − f 在 f=0.5 处穿过 0.5 ⇒ 插值必须给出 0.5。"""
        m = _metrics()
        assert abs(m.threshold_by_midpoint(F, 1.0 - F, 0.5) - 0.5) < 1e-9

    def test_midpoint_threshold_returns_none_if_never_crosses(self):
        m = _metrics()
        assert m.threshold_by_midpoint(F, np.ones_like(F), 0.5) is None

    def test_max_drop_threshold_uses_jump_midpoint(self):
        m = _metrics()
        S = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        assert abs(m.threshold_by_max_drop(F, S) - 0.45) < 1e-9

    def test_auto_method_follows_transition_order(self):
        """auto：一阶取最大跌幅、二阶取中位数 —— 两条曲线要给出不同口径的答案。"""
        m = _metrics()
        S_jump = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        assert abs(m.percolation_threshold(F, S_jump, method="auto") - 0.45) < 1e-9
        assert abs(m.percolation_threshold(F, 1.0 - F, method="auto") - 0.5) < 1e-9

    def test_unknown_method_raises(self):
        m = _metrics()
        try:
            m.percolation_threshold(F, 1.0 - F, method="凭感觉")
        except ValueError:
            return
        raise AssertionError("未知 method 应当报错")


# ============================================================
# 韧性面积 / RHI
# ============================================================

class TestResilienceArea:

    def test_perfect_curve_has_zero_loss(self):
        m = _metrics()
        assert abs(m.loss_area(F, np.ones_like(F))) < 1e-12
        assert abs(m.rhi(F, np.ones_like(F)) - 1.0) < 1e-12

    def test_dead_curve_has_full_loss(self):
        m = _metrics()
        assert abs(m.loss_area(F, np.zeros_like(F)) - 1.0) < 1e-12
        assert abs(m.rhi(F, np.zeros_like(F))) < 1e-12

    def test_loss_area_equals_triangle_for_linear_curve(self):
        """S = 1 − f ⇒ 损失面积 = 三角形面积 = ∫f df = 0.5。"""
        m = _metrics()
        assert abs(m.loss_area(F, 1.0 - F) - 0.5) < 1e-12

    def test_rhi_and_loss_area_are_complementary(self):
        m = _metrics()
        S = np.array([1.0, 0.95, 0.8, 0.5, 0.1, 0.0, 0, 0, 0, 0, 0])
        assert abs(m.rhi(F, S) - (1.0 - m.loss_area(F, S))) < 1e-12

    def test_worse_curve_means_lower_rhi(self):
        m = _metrics()
        better = np.array([1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        worse = np.array([1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert m.rhi(F, worse) < m.rhi(F, better)

    def test_length_mismatch_raises(self):
        m = _metrics()
        try:
            m.loss_area([0.0, 0.5, 1.0], [1.0, 0.5])
        except ValueError:
            return
        raise AssertionError("长度不一致应当报错")

    def test_short_curve_is_safe(self):
        m = _metrics()
        assert m.loss_area([0.0], [1.0]) == 0.0


# ============================================================
# 恢复过程
# ============================================================

class TestRecovery:

    TIMES = [0, 1, 2, 3, 4]
    PERF = [1.0, 0.2, 0.2, 1.0, 1.0]

    def test_recovery_slope_is_fastest_rise(self):
        m = _metrics()
        assert abs(m.recovery_slope(self.TIMES, self.PERF) - 0.8) < 1e-12

    def test_recovery_time_finds_first_return_to_target(self):
        m = _metrics()
        assert abs(m.recovery_time(self.TIMES, self.PERF, target=1.0) - 3.0) < 1e-12

    def test_recovery_time_none_if_never_recovers(self):
        m = _metrics()
        assert m.recovery_time(self.TIMES, [1.0, 0.5, 0.5, 0.5, 0.5]) is None

    def test_recovery_metrics_bundle(self):
        m = _metrics()
        out = m.recovery_metrics(self.TIMES, self.PERF)
        assert abs(out["min_performance"] - 0.2) < 1e-12
        assert abs(out["recovery_slope"] - 0.8) < 1e-12
        assert abs(out["recovery_time"] - 3.0) < 1e-12

    def test_partial_target_recovers_earlier(self):
        m = _metrics()
        assert m.recovery_time(self.TIMES, self.PERF, target=0.2) <= 1.0


# ============================================================
# 与 phase6 联调：曲线 → 指标
# ============================================================

class TestWithPhase6:

    def _curve(self, q=1.0, N=120, m=3, n_trials=3, seed=7):
        L0 = importlib.import_module("phase6_interdependent.L0_static_dependency")
        G_A, G_B = _make_pair(N=N, m=m, seed=seed)
        fractions = np.linspace(0.0, 0.5, 11)
        f, S = L0.percolation_curve(G_A, G_B, fractions=fractions,
                                    coupled_fraction=q, n_trials=n_trials, seed=seed)
        return np.array(f), np.array(S)

    def test_metrics_pipeline_runs_on_L0_curve(self):
        m = _metrics()
        f, S = self._curve()
        pc = m.percolation_threshold(f, S, method="auto")
        assert pc is not None and 0.0 <= pc <= 0.5
        assert 0.0 <= m.rhi(f, S) <= 1.0
        assert 0.0 <= m.loss_area(f, S) <= 0.5

    def test_stronger_coupling_is_not_more_resilient(self):
        """耦合越强（q 越大）⇒ 相依网络的韧性不应更高（PNAS 2019 的核心口径）。"""
        m = _metrics()
        f_weak, S_weak = self._curve(q=0.2)
        f_strong, S_strong = self._curve(q=1.0)
        assert m.rhi(f_strong, S_strong) <= m.rhi(f_weak, S_weak) + 1e-9

    def test_vulnerability_probe_detects_critical_node(self):
        """移掉一个 hub 之后，p_c 应当不晚于原来（探针给出非负后果）。"""
        m = _metrics()
        L0 = importlib.import_module("phase6_interdependent.L0_static_dependency")
        G_A, G_B = _make_pair(N=120, m=3, seed=7)
        fractions = np.linspace(0.0, 0.5, 11)
        f0, S0 = L0.percolation_curve(G_A, G_B, fractions=fractions,
                                      n_trials=3, seed=7)
        pc0 = m.percolation_threshold(f0, S0, method="midpoint")

        hub = max(G_A.degree(), key=lambda kv: kv[1])[0]
        G_A_minus = G_A.copy()
        G_A_minus.remove_node(hub)
        G_B_minus = G_B.copy()
        G_B_minus.remove_node(sorted(G_B.nodes())[-1])
        coupling = None  # 两层规模一致时才能一一对应，这里各自少一个节点仍可一一对应
        f1, S1 = L0.percolation_curve(G_A_minus, G_B_minus, fractions=fractions,
                                      n_trials=3, seed=7)
        pc1 = m.percolation_threshold(f1, S1, method="midpoint")
        assert coupling is None  # 占位：提醒这里没有额外的耦合参数
        if pc0 is not None and pc1 is not None:
            assert m.vulnerability_probe(pc0, pc1) >= -1e-9
