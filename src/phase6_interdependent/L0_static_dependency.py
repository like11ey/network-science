"""
Phase 6 · L0 —— 静态结构依赖（布尔耦合 + 渗流迭代 + 最大连通分量）

【这一级在干什么】
  最朴素的双层模型：节点只有"活着/失效"两种状态，层间依赖是**布尔**的
  （伙伴死了我就死），网络功能等于"我还连在巨连通分量上"。
  它是后面 L1–L3 的**基线**：L1 加自动力学、L2 加状态驱动耦合、L3 加超图，
  都必须能在极端参数下**退回 L0/L1**（见 README 的"退化性检验"纪律）。

【对拍目标】
  段老师 PNAS 2019 的 Fig.3A：ER–ER 双层网络
    · 耦合弱（q 小）→ S(f) 连续下降 = **二阶相变**（和单层网络一样温和）
    · 耦合强（q → 1）→ S(f) 突然跳落 = **一阶相变**（一崩到底）
  另外对照 Buldyrev et al., Nature 464:1025 (2010) 的经典结果。

【这一级要学的知识】
  · 渗流（percolation）：随机去掉一部分节点，看巨连通分量还剩多少
  · 相变的**阶**：连续下降是二阶；出现跳变是一级
  · 临界阈值 p_c：S 从"还很大"掉到"接近 0"的那个 f
  · 序参量 S 的用法：把整个网络的"功能"压成一个 0~1 的数

运行：
    python src/phase6_interdependent/L0_static_dependency.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .core import InterdependentNetwork

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEVEL = "L0 静态结构依赖"
__all__ = ["percolation_curve", "single_layer_curve", "transition_order", "run_experiment"]


def percolation_curve(
    G_A,
    G_B,
    fractions=None,
    coupled_fraction=1.0,
    n_trials=5,
    seed=0,
    layer="A",
):
    """
    扫一遍移除比例 f，量出每个 f 对应的存活序参量 S。

    参数:
        fractions        : 要扫的移除比例列表（默认 0 → 0.5，共 26 个点）
        coupled_fraction : 耦合强度 q，L0 的核心旋钮（q 越大越脆弱）
        n_trials         : 每个 f 重复多少次取平均（随机攻击必须多次平均）

    返回:
        (fractions, S_list)
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.5, 26)
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())

    curve = []
    for f in fractions:
        n_remove = int(round(len(nodes_A) * f))
        trials = []
        for _ in range(n_trials):
            net = InterdependentNetwork(
                G_A, G_B, coupled_fraction=coupled_fraction, seed=seed
            )
            if n_remove == 0:
                failed = []
            else:
                failed = rng.choice(nodes_A, size=n_remove, replace=False)
            trials.append(net.run(failed)["S_A" if layer == "A" else "S_B"])
        curve.append(float(np.mean(trials)))
    return list(map(float, fractions)), curve


def single_layer_curve(G, fractions=None, n_trials=5, seed=0):
    """
    单层网络的对照曲线（没有层间依赖时会长什么样）。

    做法：随机移除 f 比例的节点，只保留最大连通分量，量 GCC/N。
    这正是"二阶相变"的典型形状，用来和双层的一阶跳变对照。
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.5, 26)
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    N = len(nodes)

    curve = []
    for f in fractions:
        n_remove = int(round(N * f))
        trials = []
        for _ in range(n_trials):
            if n_remove == 0:
                remain = nodes
            else:
                removed = set(rng.choice(nodes, size=n_remove, replace=False))
                remain = [n for n in nodes if n not in removed]
            if not remain:
                trials.append(0.0)
                continue
            sub = G.subgraph(remain)
            trials.append(len(max(nx.connected_components(sub), key=len)) / N)
        curve.append(float(np.mean(trials)))
    return list(map(float, fractions)), curve


def transition_order(fractions, S, jump_threshold=0.2):
    """
    判断相变是"一阶"还是"二阶"——靠 S 曲线的**最大单步跌幅**。

    判据（本项目的操作定义，汇报时说清楚口径）：
        最大单步跌幅 > jump_threshold  →  '一阶'（有跳变，突然崩塌）
        否则                            →  '二阶'（连续下降）

    返回:
        (order_str, max_jump)
    """
    S = np.asarray(S, dtype=float)
    if len(S) < 2:
        return "数据不足", 0.0
    max_jump = float(np.max(np.abs(np.diff(S))))
    return ("一阶" if max_jump > jump_threshold else "二阶"), max_jump


def run_experiment(N=300, m=3, n_trials=5, seed=42, save=True):
    """跑 L0 的标准实验：弱耦合 vs 强耦合 vs 单层，并画在一张图上。"""
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    fractions = np.linspace(0.0, 0.5, 26)

    _, s_single = single_layer_curve(G_A, fractions, n_trials=n_trials, seed=seed)
    _, s_weak = percolation_curve(
        G_A, G_B, fractions, coupled_fraction=0.2, n_trials=n_trials, seed=seed
    )
    _, s_strong = percolation_curve(
        G_A, G_B, fractions, coupled_fraction=1.0, n_trials=n_trials, seed=seed
    )

    order_weak, jump_weak = transition_order(fractions, s_weak)
    order_strong, jump_strong = transition_order(fractions, s_strong)

    print(f"=== {LEVEL}：耦合强度决定相变阶数 ===")
    print(f"  弱耦合 q=0.2 ：相变={order_weak}（最大单步跌幅 {jump_weak:.3f}）")
    print(f"  强耦合 q=1.0 ：相变={order_strong}（最大单步跌幅 {jump_strong:.3f}）")

    if save:
        plt.figure(figsize=(9, 6))
        plt.plot(fractions, s_single, "o-", label="单层网络（对照）", color="tab:blue")
        plt.plot(fractions, s_weak, "s-", label="双层 · 弱耦合 q=0.2", color="tab:green")
        plt.plot(fractions, s_strong, "^-", label="双层 · 强耦合 q=1.0", color="tab:red")
        plt.xlabel("移除节点比例 f")
        plt.ylabel("存活序参量 S（GCC 占比）")
        plt.title(f"{LEVEL}：相依网络的脆弱性来自层间耦合")
        plt.legend()
        plt.grid(alpha=0.3)
        out = RESULTS_DIR / "phase6_L0_percolation.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图已保存：{out}")

    return {
        "fractions": list(fractions),
        "S_single": s_single,
        "S_weak": s_weak,
        "S_strong": s_strong,
        "order_weak": order_weak,
        "order_strong": order_strong,
    }


if __name__ == "__main__":
    run_experiment()
