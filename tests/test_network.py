"""
测试文件 - 学习Python测试框架（pytest）
"""

import sys
import os
import importlib

# 把 src 目录加入 Python 路径（用绝对路径，兼容任何安装位置）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import networkx as nx
import numpy as np
from collections import Counter


def _import_from(subpackage, module_filename, name):
    """从 src/ 下的子包中导入指定名称。
    因为文件名以数字开头(如 01_four_models)，不能用常规 import 语法。"""
    module = importlib.import_module(f"{subpackage}.{module_filename}")
    return getattr(module, name)


def test_er_network_avg_degree():
    """测试ER网络平均度"""
    N, p = 500, 0.02
    G = nx.erdos_renyi_graph(N, p, seed=42)
    avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    assert abs(avg_degree - N * p) < 3


def test_ba_network_scale_free():
    """测试BA网络的幂律特性"""
    G = nx.barabasi_albert_graph(200, 3, seed=42)
    degrees = [d for _, d in G.degree()]
    avg_deg = np.mean(degrees)
    max_deg = max(degrees)
    assert max_deg > avg_deg * 3


def test_regular_network_properties():
    """测试规则网络属性"""
    create_regular_network = _import_from(
        "phase3_network_models", "01_four_models", "create_regular_network"
    )
    G = create_regular_network(50, 6)
    degrees = [d for _, d in G.degree()]
    assert len(set(degrees)) == 1
    assert degrees[0] == 6


def test_ba_robust_to_random():
    """BA网络对随机故障鲁棒"""
    random_removal = _import_from(
        "phase4_robustness", "01_robustness_analysis", "random_removal"
    )
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    G_removed = random_removal(G, 0.1)
    largest_cc = max(nx.connected_components(G_removed), key=len)
    giant_ratio = len(largest_cc) / G_removed.number_of_nodes()
    assert giant_ratio > 0.5


def test_ba脆弱_to_targeted():
    """BA网络对蓄意攻击脆弱"""
    targeted_removal_by_degree = _import_from(
        "phase4_robustness", "01_robustness_analysis", "targeted_removal_by_degree"
    )
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    G_removed = targeted_removal_by_degree(G, 0.2)
    largest_cc = max(nx.connected_components(G_removed), key=len)
    giant_ratio = len(largest_cc) / G_removed.number_of_nodes()
    assert giant_ratio < 0.8


def test_cascade_simulator_init():
    """测试仿真器初始化"""
    CascadingFailureSimulator = _import_from(
        "phase5_cascading", "01_cascading_failure", "CascadingFailureSimulator"
    )
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    sim = CascadingFailureSimulator(G, alpha=0.5)
    assert sim.N == 50
    assert len(sim.loads) == 50
    assert len(sim.capacities) == 50


def test_capacity_parameter():
    """测试容量参数"""
    CascadingFailureSimulator = _import_from(
        "phase5_cascading", "01_cascading_failure", "CascadingFailureSimulator"
    )
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    sim_small = CascadingFailureSimulator(G, alpha=0.1)
    sim_large = CascadingFailureSimulator(G, alpha=0.5)
    _, failed_small, _ = sim_small.simulate_single_attack(0)
    _, failed_large, _ = sim_large.simulate_single_attack(0)
    assert failed_small >= failed_large


def test_interdependent_creation():
    """测试相依网络创建"""
    InterdependentNetwork = _import_from(
        "phase6_interdependent", "01_interdependent_cascading", "InterdependentNetwork"
    )
    G_A = nx.erdos_renyi_graph(30, 0.05, seed=42)
    G_B = nx.erdos_renyi_graph(30, 0.05, seed=43)
    net = InterdependentNetwork(G_A, G_B)
    assert net.N_A == 30
    assert net.N_B == 30
    assert len(net.coupling) == 30


if __name__ == "__main__":
    print("运行测试...")
    print("请安装pytest: pip install pytest")
    print("然后运行: pytest tests/test_network.py -v")
