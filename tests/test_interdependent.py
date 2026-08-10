"""
tests/test_interdependent.py - Phase 6 相依网络级联失效 针对性测试

覆盖核心类和方法：
- InterdependentNetwork: 初始化、耦合构建、重置、级联仿真、断连移除
"""

import sys
import os
import importlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import networkx as nx
import numpy as np


def _import_from(subpackage, module_filename, name):
    module = importlib.import_module(f"{subpackage}.{module_filename}")
    return getattr(module, name)


def _get_interdependent_class():
    return _import_from(
        "phase6_interdependent", "01_interdependent_cascading", "InterdependentNetwork"
    )


# ============================================================
# InterdependentNetwork 初始化与耦合
# ============================================================

class TestInterdependentNetworkInit:
    """测试相依网络初始化和耦合关系"""

    def test_node_counts_preserved(self):
        """两层网络的节点数应正确保留"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(50, 3, seed=42)
        G_B = nx.barabasi_albert_graph(60, 3, seed=43)
        net = IN(G_A, G_B)
        assert net.N_A == 50
        assert net.N_B == 60

    def test_one_to_one_coupling_count(self):
        """一一对应耦合的数量 = min(N_A, N_B)"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(30, 3, seed=42)
        G_B = nx.barabasi_albert_graph(30, 3, seed=43)
        net = IN(G_A, G_B, coupling_type="one_to_one")
        assert len(net.coupling) == 30

    def test_one_to_one_coupling_is_bijective(self):
        """一一对应耦合中，每个A节点映射到唯一的B节点"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        net = IN(G_A, G_B, coupling_type="one_to_one")
        # 值集合应无重复（双射）
        coupled_B = list(net.coupling.values())
        assert len(set(coupled_B)) == len(coupled_B)

    def test_random_coupling_count(self):
        """随机耦合的数量 = N_A"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(25, 3, seed=42)
        G_B = nx.barabasi_albert_graph(25, 3, seed=43)
        net = IN(G_A, G_B, coupling_type="random")
        assert len(net.coupling) == 25

    def test_original_graphs_not_modified(self):
        """构造不应修改原始图"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        orig_A_edges = G_A.number_of_edges()
        orig_B_edges = G_B.number_of_edges()
        _ = IN(G_A, G_B)
        assert G_A.number_of_edges() == orig_A_edges
        assert G_B.number_of_edges() == orig_B_edges

    def test_all_nodes_initially_alive(self):
        """初始化后所有节点应存活"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        net = IN(G_A, G_B)
        assert all(net.alive_A.values())
        assert all(net.alive_B.values())


# ============================================================
# reset 方法
# ============================================================

class TestReset:
    """测试重置功能"""

    def test_reset_restores_all_alive(self):
        """reset 后所有节点应恢复存活"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        net = IN(G_A, G_B)
        # 先模拟一次级联
        net.simulate_cascade([0, 1])
        # 重置
        net.reset()
        assert all(net.alive_A.values())
        assert all(net.alive_B.values())


# ============================================================
# simulate_cascade 级联仿真
# ============================================================

class TestCascadeSimulation:
    """测试相依网络级联仿真的行为不变量"""

    def test_no_initial_failures_all_survive(self):
        """无初始失效时，所有节点应存活"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(30, 3, seed=42)
        G_B = nx.barabasi_albert_graph(30, 3, seed=43)
        net = IN(G_A, G_B)
        result = net.simulate_cascade([])
        assert result["surviving_A"] == 1.0
        assert result["surviving_B"] == 1.0
        assert result["total_surviving"] == 1.0

    def test_all_A_removed_zero_surviving_A(self):
        """移除所有A层节点后，A层存活比例为0"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        net = IN(G_A, G_B)
        all_nodes = list(G_A.nodes())
        result = net.simulate_cascade(all_nodes)
        assert result["surviving_A"] == 0.0

    def test_result_keys_present(self):
        """返回字典应包含所有预期键"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(20, 3, seed=42)
        G_B = nx.barabasi_albert_graph(20, 3, seed=43)
        net = IN(G_A, G_B)
        result = net.simulate_cascade([0])
        expected_keys = {"surviving_A", "surviving_B", "total_surviving", "failed_A", "failed_B"}
        assert set(result.keys()) == expected_keys

    def test_surviving_and_failed_sum_to_one(self):
        """每层存活比例 + 失效比例 = 1.0"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(30, 3, seed=42)
        G_B = nx.barabasi_albert_graph(30, 3, seed=43)
        net = IN(G_A, G_B)
        result = net.simulate_cascade([0, 1, 2])
        assert abs(result["surviving_A"] + result["failed_A"] - 1.0) < 1e-10
        assert abs(result["surviving_B"] + result["failed_B"] - 1.0) < 1e-10

    def test_cascade_reduces_surviving(self):
        """级联后存活比例应 <= 1.0"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(50, 3, seed=42)
        G_B = nx.barabasi_albert_graph(50, 3, seed=43)
        net = IN(G_A, G_B)
        result = net.simulate_cascade([0])
        assert result["surviving_A"] <= 1.0
        assert result["surviving_B"] <= 1.0

    def test_targeted_attack_worse_than_random(self):
        """蓄意攻击（移除hub）通常比随机移除导致更多失效"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(50, 3, seed=42)
        G_B = nx.barabasi_albert_graph(50, 3, seed=43)

        # 蓄意攻击：移除度最大的节点
        net_targeted = IN(G_A, G_B)
        hub = max(G_A.degree(), key=lambda x: x[1])[0]
        result_targeted = net_targeted.simulate_cascade([hub])

        # 随机移除：移除一个叶子节点（度最小）
        net_random = IN(G_A, G_B)
        leaf = min(G_A.degree(), key=lambda x: x[1])[0]
        result_random = net_random.simulate_cascade([leaf])

        # hub攻击的存活率应 <= 叶子攻击
        assert result_targeted["total_surviving"] <= result_random["total_surviving"] + 0.01

    def test_values_in_valid_range(self):
        """所有返回值应在 [0, 1] 范围内"""
        IN = _get_interdependent_class()
        G_A = nx.barabasi_albert_graph(30, 3, seed=42)
        G_B = nx.barabasi_albert_graph(30, 3, seed=43)
        net = IN(G_A, G_B)
        result = net.simulate_cascade([0, 1])
        for key in result:
            assert 0 <= result[key] <= 1.0


# ============================================================
# _remove_disconnected 断连移除
# ============================================================

class TestRemoveDisconnected:
    """测试断连节点移除逻辑"""

    def test_connected_graph_no_removal(self):
        """连通图不应移除任何节点"""
        IN = _get_interdependent_class()
        G = nx.barabasi_albert_graph(20, 3, seed=42)
        alive = {n: True for n in G.nodes()}
        removed = net = IN(G, G)
        removed_set = net._remove_disconnected(G, alive)
        assert len(removed_set) == 0

    def test_all_dead_returns_empty(self):
        """所有节点死亡时返回空集"""
        IN = _get_interdependent_class()
        G = nx.path_graph(5)
        alive = {n: False for n in G.nodes()}
        net = IN(G, G)
        removed = net._remove_disconnected(G, alive)
        assert len(removed) == 0

    def test_isolated_node_removed(self):
        """与最大连通分量分离的节点应被移除"""
        IN = _get_interdependent_class()
        # 创建路径图 0-1-2-3-4
        G = nx.path_graph(5)
        alive = {n: True for n in G.nodes()}
        # 使节点2死亡，将图分为两部分: {0,1} 和 {3,4}
        alive[2] = False
        net = IN(G, G)
        removed = net._remove_disconnected(G, alive)
        # 较小连通分量中的节点应被移除
        assert len(removed) > 0
        # 被移除的节点应标记为死亡
        for n in removed:
            assert alive[n] is False
