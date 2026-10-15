"""
tests/test_interdependent.py — Phase 6 四级台阶（L0→L3）针对性测试

测试分组：
- TestCoreSkeleton : core.py 骨架（耦合构造 / 记账 / 级联循环 / GCC 剪枝）
- TestL0Static     : L0 静态结构依赖（渗流曲线、相变阶数、与单层对比）
- TestL1Dynamics   : L1 网内自动力学（容忍判据 δ，**δ=1 必须退回 L0**）
- TestL2StateDriven: L2 状态驱动耦合 W(t)（**ρ=0 且 φ=0 必须退回 L0**）
- TestL3Hypergraph : L3 超图逻辑依赖（AND/OR/k-of-N，**二元 AND 必须退回 L0**）

三条退化性检验是本文件的重点：一个四级模型必须能证明自己**包含**上一级，
否则"更真实"就只是多加了旋钮而已。
"""

import os
import sys
import importlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import networkx as nx
import numpy as np


def _load(module_filename):
    """按文件名加载 src/phase6_interdependent/ 下的模块（沿用本项目既有约定）。"""
    return importlib.import_module(f"phase6_interdependent.{module_filename}")


def _make_pair(N=60, m=2, seed=42):
    """造一对 BA 网络（两层规模/生成方式相同，便于对拍）。"""
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    return G_A, G_B


def _S_curve(rule_of, fractions, net_cls=None, N=60, m=2, seed=42, n_trials=2):
    """
    用**同一批网络 + 同一批初始故障**扫 f，得平均 S_A 曲线。

    退化性检验只能这样比：种子、初始故障序列、网络都必须完全一致，
    否则"曲线略有差别"到底是模型差异还是随机噪声就说不清了。
    """
    core = _load("core")
    net_cls = net_cls or core.InterdependentNetwork
    G_A, G_B = _make_pair(N=N, m=m, seed=seed)
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())

    curve = []
    for f in fractions:
        n_remove = int(round(len(nodes_A) * f))
        trials = []
        for _ in range(n_trials):
            net = net_cls(G_A, G_B, seed=seed)
            failed = ([] if n_remove == 0
                      else rng.choice(nodes_A, size=n_remove, replace=False))
            trials.append(net.run(failed, rule=rule_of(net))["S_A"])
        curve.append(float(np.mean(trials)))
    return curve


FAIR_FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3]


# ============================================================
# core.py 骨架
# ============================================================

class TestCoreSkeleton:

    def test_node_counts_preserved(self):
        core = _load("core")
        G_A, G_B = _make_pair(N=50, m=3)
        net = core.InterdependentNetwork(G_A, G_B)
        assert net.n_nodes("A") == 50
        assert net.n_nodes("B") == 50

    def test_one_to_one_coupling_is_bijective(self):
        core = _load("core")
        G_A, G_B = _make_pair(N=40, m=2)
        net = core.InterdependentNetwork(G_A, G_B, coupling="one_to_one")
        fwd, bwd = net.partner["A"], net.partner["B"]
        assert len(fwd) == 40 and len(bwd) == 40
        assert len(set(fwd.values())) == len(fwd)          # 一对一无重复
        for a, b in fwd.items():
            assert bwd[b] == a                             # 反向查得到

    def test_coupled_fraction_controls_coupling_count(self):
        """耦合比例 q 是 PNAS 2019 图 3A 的旋钮：q 小 ⇒ 只有少数节点互相依赖。"""
        core = _load("core")
        G_A, G_B = _make_pair(N=60, m=2)
        net = core.InterdependentNetwork(G_A, G_B, coupled_fraction=0.5)
        assert len(net.partner["A"]) == 30

    def test_random_mode_same_count(self):
        core = _load("core")
        G_A, G_B = _make_pair(N=40, m=2)
        net = core.InterdependentNetwork(G_A, G_B, coupling="random", seed=7)
        assert len(net.partner["A"]) == 40

    def test_original_graphs_not_modified(self):
        core = _load("core")
        G_A, G_B = _make_pair(N=30, m=2)
        e_A, e_B = G_A.number_of_edges(), G_B.number_of_edges()
        core.InterdependentNetwork(G_A, G_B)
        assert (G_A.number_of_edges(), G_B.number_of_edges()) == (e_A, e_B)

    def test_fresh_network_all_alive_and_full_state(self):
        core = _load("core")
        G_A, G_B = _make_pair()
        net = core.InterdependentNetwork(G_A, G_B)
        assert all(net.alive["A"].values()) and all(net.alive["B"].values())
        assert all(v == 1.0 for v in net.x["A"].values())

    def test_kill_zeroes_state(self):
        core = _load("core")
        G_A, G_B = _make_pair()
        net = core.InterdependentNetwork(G_A, G_B)
        net.kill("A", 0)
        assert net.is_alive("A", 0) is False
        assert net.x["A"][0] == 0.0

    def test_reset_restores_everything(self):
        core = _load("core")
        G_A, G_B = _make_pair()
        net = core.InterdependentNetwork(G_A, G_B)
        net.run([0, 1, 2])
        net.reset()
        assert all(net.alive["A"].values()) and all(net.alive["B"].values())

    def test_no_failures_means_no_cascade(self):
        core = _load("core")
        G_A, G_B = _make_pair()
        net = core.InterdependentNetwork(G_A, G_B)
        res = net.run([])
        assert res["S_A"] == 1.0 and res["S_B"] == 1.0
        assert res["steps"] == 0

    def test_partner_failure_propagates_across_layers(self):
        """布尔耦合的核心行为：伙伴死了我也死（这就是层间依赖）。"""
        core = _load("core")
        G_A, G_B = _make_pair(N=40, m=2)
        net = core.InterdependentNetwork(G_A, G_B)
        victim = sorted(G_A.nodes())[5]
        partner = net.partner["A"][victim]
        net.run([victim])
        assert net.is_alive("B", partner) is False

    def test_removing_all_of_layer_a_kills_layer_b(self):
        core = _load("core")
        G_A, G_B = _make_pair(N=30, m=2)
        net = core.InterdependentNetwork(G_A, G_B)
        res = net.run(list(G_A.nodes()))
        assert res["S_A"] == 0.0 and res["S_B"] == 0.0

    def test_run_returns_documented_keys(self):
        core = _load("core")
        G_A, G_B = _make_pair()
        net = core.InterdependentNetwork(G_A, G_B)
        res = net.run([0], record=True)
        assert set(res.keys()) == {"S_A", "S_B", "S_mean", "steps",
                                   "failed_initial", "history"}
        assert res["history"][0]["step"] == 0

    def test_prune_gcc_kills_disconnected_nodes(self):
        """与主干断开的节点即使"物理完好"也算失去功能。"""
        core = _load("core")
        G = nx.path_graph(7)
        net = core.InterdependentNetwork(G, G)
        net.kill("A", 3)                     # 路径被切成 {0,1,2} 与 {4,5,6}
        pruned = net.prune_gcc("A")
        assert pruned                            # 较小的一侧被剪掉
        assert net.gcc("A") == {4, 5, 6} or net.gcc("A") == {0, 1, 2}

    def test_build_coupling_returns_both_directions(self):
        core = _load("core")
        coupling = core.build_coupling([0, 1, 2], [10, 11, 12], seed=0)
        assert coupling["A"][0] == 10 and coupling["B"][10] == 0


# ============================================================
# L0 静态结构依赖
# ============================================================

class TestL0Static:

    def test_percolation_curve_is_non_increasing(self):
        """移除越多，存活越少 —— 曲线必须单调不增。"""
        L0 = _load("L0_static_dependency")
        G_A, G_B = _make_pair(N=100, m=3)
        fractions = np.linspace(0.0, 0.5, 11)
        # 注意返回形状：(fractions, S_list)
        _, S = L0.percolation_curve(G_A, G_B, fractions=fractions, n_trials=5, seed=1)
        assert S[0] == 1.0                      # 不攻击 ⇒ 全网存活
        assert S[-1] <= S[0] - 0.2              # 攻击 f=0.5 ⇒ 明显崩塌
        # 随机攻击有抽样噪声，所以只要求"总体不增"（允许 0.1 的抖动）
        assert all(S[i] >= S[i + 1] - 0.1 for i in range(len(S) - 1))

    def test_transition_order_flags_a_jump_as_first_order(self):
        L0 = _load("L0_static_dependency")
        fractions = np.linspace(0, 1, 11)
        jumping = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        order, jump = L0.transition_order(fractions, jumping)
        assert order == "一阶" and jump > 0.9

    def test_transition_order_flags_smooth_decline_as_second_order(self):
        L0 = _load("L0_static_dependency")
        fractions = np.linspace(0, 1, 11)
        smooth = list(np.linspace(1.0, 0.0, 11))
        order, jump = L0.transition_order(fractions, smooth)
        assert order == "二阶" and jump < 0.2

    def test_interdependent_is_more_fragile_than_single_layer(self):
        """PNAS 2019 的立论：相依网络比单层网络脆弱得多。"""
        L0 = _load("L0_static_dependency")
        G_A, G_B = _make_pair(N=150, m=3)
        fractions = np.array([0.0, 0.1, 0.2])
        _, S_inter = L0.percolation_curve(G_A, G_B, fractions=fractions,
                                          n_trials=3, seed=5)
        _, S_single = L0.single_layer_curve(G_A, fractions, n_trials=3, seed=5)
        assert S_inter[-1] <= S_single[-1] + 1e-9

    def test_run_experiment_returns_curve_data(self):
        L0 = _load("L0_static_dependency")
        out = L0.run_experiment(N=60, m=2, n_trials=1, seed=3, save=False)
        assert set(out.keys()) >= {"fractions", "S_single", "S_weak", "S_strong",
                                   "order_weak", "order_strong"}
        assert len(out["fractions"]) == len(out["S_strong"])


# ============================================================
# L1 网内自动力学（δ 容忍带）
# ============================================================

class TestL1Dynamics:

    def _cfg_and_ref(self, delta):
        L1 = _load("L1_intra_layer_dynamics")
        G_A, G_B = _make_pair(N=60, m=2)
        cfg = L1.AutoDynamicsConfig(mu=0.4, lam=0.3, omega=0.3, delta=delta)
        x_ref = L1.healthy_reference(G_A, G_B, cfg)
        return L1, cfg, x_ref

    def test_healthy_reference_is_one(self):
        L1, cfg, x_ref = self._cfg_and_ref(delta=0.2)
        assert all(abs(v - 1.0) < 1e-9 for v in x_ref["A"].values())
        assert set(x_ref.keys()) == {"A", "B"}

    def test_state_drops_when_partner_dies(self):
        L1, cfg, x_ref = self._cfg_and_ref(delta=0.2)
        core = _load("core")
        G_A, G_B = _make_pair(N=60, m=2)
        net = core.InterdependentNetwork(G_A, G_B)
        victim = sorted(G_A.nodes())[0]
        net.kill("A", victim)
        L1.run_dynamics(net, cfg)
        assert net.x["A"][victim] == 0.0
        assert all(v <= 1.0 + 1e-9 for v in net.x["B"].values())

    def test_delta_one_degenerates_to_L0(self):
        """⭐ 退化性检验：δ=1 ⇒ 容忍带最宽 ⇒ 无动力学失效 ⇒ 必须等于 L0。"""
        L1, cfg_one, x_ref = self._cfg_and_ref(delta=1.0)
        curve_L1 = _S_curve(lambda net: L1.L1Rule(cfg_one, x_ref), FAIR_FRACTIONS)
        curve_L0 = _S_curve(lambda net: None, FAIR_FRACTIONS)
        assert np.allclose(curve_L1, curve_L0, atol=1e-9)

    def test_small_delta_is_not_more_robust(self):
        """δ 越小 ⇒ 容忍带越窄 ⇒ 级联不应更轻（越大越好只可能变差）。"""
        L1, cfg_small, x_ref_small = self._cfg_and_ref(delta=0.05)
        L1b, cfg_one, x_ref_one = self._cfg_and_ref(delta=1.0)
        curve_small = _S_curve(lambda net: L1.L1Rule(cfg_small, x_ref_small),
                               FAIR_FRACTIONS)
        curve_one = _S_curve(lambda net: L1b.L1Rule(cfg_one, x_ref_one),
                             FAIR_FRACTIONS)
        at_f = -1
        assert curve_small[at_f] <= curve_one[at_f] + 1e-9


# ============================================================
# L2 状态驱动耦合 W(t)
# ============================================================

class TestL2StateDriven:

    def test_rho_zero_keeps_weight_constant(self):
        L2 = _load("L2_state_driven_coupling")
        cfg = L2.StateDrivenConfig(w0=0.5, rho=0.0)
        assert abs(L2.coupling_weight(1.0, 0.2, cfg) - 0.5) < 1e-12
        assert abs(L2.coupling_weight(0.0, 0.0, cfg) - 0.5) < 1e-12

    def test_worse_state_means_weaker_delivered_coupling(self):
        """状态越差 ⇒ 跨层供过来的越少（w 单调递减）；ρ=0 ⇒ 恒为 w0。"""
        L2 = _load("L2_state_driven_coupling")
        cfg = L2.StateDrivenConfig(w0=1.0, rho=1.0)
        w_healthy = L2.coupling_weight(1.0, 1.0, cfg)
        w_degraded = L2.coupling_weight(0.4, 1.0, cfg)
        w_dead = L2.coupling_weight(0.0, 1.0, cfg)
        assert w_healthy > w_degraded > w_dead
        assert abs(w_healthy - 1.0) < 1e-12  # 满血不打折 ⇒ 健康不动点为 1
        cfg0 = L2.StateDrivenConfig(w0=1.0, rho=0.0)
        assert abs(L2.coupling_weight(0.0, 1.0, cfg0) - 1.0) < 1e-12

    def test_zero_rho_zero_phi_degenerates_to_L0(self):
        """⭐ 退化性检验：W ≡ w0（ρ=0）且 φ=0 ⇒ 无动力学失效 ⇒ 必须等于 L0。"""
        L2 = _load("L2_state_driven_coupling")
        cfg = L2.StateDrivenConfig(w0=1.0, rho=0.0, phi=0.0)
        curve_L2 = _S_curve(lambda net: L2.L2Rule(cfg), FAIR_FRACTIONS)
        curve_L0 = _S_curve(lambda net: None, FAIR_FRACTIONS)
        assert np.allclose(curve_L2, curve_L0, atol=1e-9)

    def test_satisfaction_threshold_only_hurts(self):
        """φ 越大（服务质量要求越高）⇒ 存活不应更多。"""
        L2 = _load("L2_state_driven_coupling")
        cfg_strict = L2.StateDrivenConfig(rho=1.0, phi=0.6)
        cfg_loose = L2.StateDrivenConfig(rho=1.0, phi=0.0)
        curve_strict = _S_curve(lambda net: L2.L2Rule(cfg_strict), FAIR_FRACTIONS)
        curve_loose = _S_curve(lambda net: L2.L2Rule(cfg_loose), FAIR_FRACTIONS)
        assert curve_strict[-1] <= curve_loose[-1] + 1e-9


# ============================================================
# L3 超图逻辑依赖
# ============================================================

class TestL3Hypergraph:

    def test_activation_rules(self):
        L3 = _load("L3_hypergraph_logic")
        alive = {"A": {0: True, 1: True, 2: False}, "B": {0: True}}
        members = [("A", 0), ("A", 1), ("A", 2)]
        assert L3.Hyperedge(members, "AND").is_active(alive) is False
        assert L3.Hyperedge(members, "OR").is_active(alive) is True
        assert L3.Hyperedge(members, "k-of-N", k=2).is_active(alive) is True
        assert L3.Hyperedge(members, "k-of-N", k=3).is_active(alive) is False

    def test_functional_measure_is_one_when_healthy(self):
        L3 = _load("L3_hypergraph_logic")
        G_A, G_B = _make_pair(N=40, m=2)
        units = L3.random_functional_hyperedges(
            L3.HyperedgeNetwork(G_A, G_B, seed=0), n_edges=10, size=3, seed=1)
        net = L3.HyperedgeNetwork(G_A, G_B, hyperedges=units, seed=0)
        assert net.functional_measure() == 1.0

    def test_pair_hyperedges_degenerate_to_L0(self):
        """⭐ 退化性检验：只用二元 AND 超边（等价成对耦合）⇒ 必须等于 L0。"""
        L3 = _load("L3_hypergraph_logic")
        curve_L3 = _S_curve(lambda net: net.hyperedge_rule(),
                            FAIR_FRACTIONS,
                            net_cls=L3.HyperedgeNetwork)
        curve_L0 = _S_curve(lambda net: None, FAIR_FRACTIONS)
        assert np.allclose(curve_L3, curve_L0, atol=1e-9)

    def test_and_is_stricter_than_or(self):
        """同一批功能单元 + 同一次攻击：AND 语义的功能度量不应高于 OR。"""
        L3 = _load("L3_hypergraph_logic")
        G_A, G_B = _make_pair(N=60, m=2)
        template = L3.HyperedgeNetwork(G_A, G_B, seed=0)
        units = L3.random_functional_hyperedges(template, n_edges=20, size=3, seed=1)
        and_units = [L3.Hyperedge(list(e.members), "AND") for e in units]
        or_units = [L3.Hyperedge(list(e.members), "OR") for e in units]

        failed = sorted(G_A.nodes())[:6]
        net_and = L3.HyperedgeNetwork(G_A, G_B, hyperedges=and_units, seed=0)
        net_and.run(failed, rule=net_and.hyperedge_rule())
        net_or = L3.HyperedgeNetwork(G_A, G_B, hyperedges=or_units, seed=0)
        net_or.run(failed, rule=net_or.hyperedge_rule())
        assert net_and.functional_measure() <= net_or.functional_measure() + 1e-9

    def test_functional_measure_drops_after_attack(self):
        L3 = _load("L3_hypergraph_logic")
        G_A, G_B = _make_pair(N=60, m=2)
        template = L3.HyperedgeNetwork(G_A, G_B, seed=0)
        units = L3.random_functional_hyperedges(template, n_edges=20, size=3, seed=1)
        net = L3.HyperedgeNetwork(G_A, G_B, hyperedges=units, seed=0)
        before = net.functional_measure()
        net.run(sorted(G_A.nodes())[:20], rule=net.hyperedge_rule())
        assert net.functional_measure() < before

    def test_run_experiment_returns_both_measures(self):
        L3 = _load("L3_hypergraph_logic")
        out = L3.run_experiment(N=60, m=2, n_trials=1, seed=3, save=False)
        assert set(out["curves"].keys()) == {"AND（全活才工作）",
                                            "k-of-N（≥2 台活着）",
                                            "OR（任一活着就工作）"}
        for d in out["curves"].values():
            assert len(d["S"]) == len(d["F"]) == len(out["fractions"])
