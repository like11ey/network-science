"""
Phase 6 · L2 —— 加状态驱动耦合 W(t)（耦合强度随状态演化 + 满足度判据 Φ < φ）

【L1 缺了什么】
  L1 里层间耦合强度是**常数 ω**：不管两边状态如何，跨层影响永远一样大。
  现实中耦合是**动态**的：
    · 电力的负载越重 → 通信/控制通道的压力越大 → 耦合越"黏"；
    · 一个节点性能掉下去，它的伙伴要额外分担，**拖累反而被放大**。
  于是耦合权重写成随状态变化的时间函数 **W(t)**——这就是本项目研究内容 1
  里"动态耦合"的第一层含义（也是自研创新点 CSQ1 的落点）。

【本级的模型（本项目口径，汇报时要说明）】
  1. 耦合权重（边上的量，随时间变）：
         w_ij(t) = w0 / [1 + ρ · (1 − min(x_i(t), x_j(t)))]
     含义：两边都满血时 w = w0（不打折）；任一方掉到 0 时 w = w0/(1+ρ)。
     物理读法：**状态越差，跳层能供过来的越少**（耦合通道自己也跟着退化）。
     ρ = 0 ⇒ W ≡ w0 ⇒ **常数耦合，退回 L1**（退化性检验用）。
  2. 供给与需求（增量式，与 L1 同一骨架 + W(t)）：
         x_i(t+1) = x_i(t)
                  + μ·(1 − x_i(t))                            ← 自反馈回弹（与 L1 的 μ 同义）
                  + (1−ω−μ)·( Σ_{存活 j} x_j(t) / k_i⁰ − x_i(t) )  ← 网内供给
                  + ω·( w_ij(t)·x_partner(t) − x_i(t) )            ← 跨层供给（带 W(t) 打折）
         需求 d_i     = 1（归一化；将来可按节点重要度设成 d_i ≠ 1）
  3. 满足度判据：
         Φ_i(t) = s_i(t) / d_i ；  Φ_i(t) < φ  →  该节点失去功能
     φ 是"服务质量底线"：φ=0 表示再差也能凑合（⇒ 无动力学失效）。
  4. 布尔耦合 + GCC 剪枝照旧兜底。

     ⚠️ 写法上的两个坑（2026-10-15 修，与 L1 同源）：
       · 耦合项必须是**打折（除以 1+ρ…）**而不是放大：写放大等于给退化节点
         "补贴"更多供给，机制反而变成稳定化，与"状态差 → 更容易崩"的假设相反。
       · 网内项按**原始度 k_i⁰** 归一（死邻居贡献 0，不被重新归一化抹平），
         且**没有伙伴的节点不吃 ω 项**（否则健康网自己就先不合格）。
       · 必须有**自反馈回弹项 μ**（与 L1 同源）：只写 (1−ω)·供给 + ω·跨层 时，
         一个"没有同层邻居"的孤立节点只能拿到 ω·伙伴，反过来把伙伴也拖下去，
         逐层互拖 ⇒ 健康网在 φ 稍大时就整网归零。

【⭐ 退化性检验】
  ρ = 0（W ≡ w0）且 φ = 0 ⇒ 本级必须**完全退回 L0**（无动力学失效）。
  见 tests/test_interdependent.py::test_L2_degenerates_to_L0。

【这一级要学的知识】
  · 状态依赖网络（state-dependent network）：拓扑/权重本身是状态的函数
  · 时间函数 W(t) 怎么写进代码：每轮迭代重新算一遍权重（不要缓存！）
  · 满足度/服务质量阈值：从"物理连通"到"功能可用"的第二层判据
  · 参数退化的意义：一个三级模型能证明自己包含上一级，才有资格谈"更真实"

⚠️ 符号提醒：PNAS 2019 原文里 W(x_i) 表示**自动力学**，
   本项目把自动力学统一记作 M(x_i)，W(t) 专门留给这里的**耦合张量**，避免撞名。

运行：
    python src/phase6_interdependent/L2_state_driven_coupling.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:  # 作为包被导入
    from .core import InterdependentNetwork, boolean_coupling_rule
except ImportError:  # 直接当脚本跑
    from core import InterdependentNetwork, boolean_coupling_rule

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEVEL = "L2 状态驱动耦合 W(t)"
__all__ = ["StateDrivenConfig", "coupling_weight", "L2Rule", "run_experiment"]


class StateDrivenConfig:
    """
    L2 的旋钮。

    w0 : 基础耦合强度（满血且 ρ=0 时 W ≡ w0；健康不动点=1 要求 w0=1）
    rho: 耦合退化系数 ρ —— 状态越差，跨层供给打折越狠（ρ=0 ⇒ 退回 L1）
    phi: 服务质量底线 φ —— 满足度低于它就停机（φ=0 ⇒ 退回 L1）
    ω  : 跨层影响的相对权重（与 L1 的 ω 同义）
    mu : 自反馈回弹强度（与 L1 的 μ 同义；网内权重 = 1−ω−μ）
    """

    def __init__(self, w0=1.0, rho=1.0, phi=0.3, omega=0.3, mu=0.4, max_iter=30, tol=1e-6):
        self.mu = mu
        self.w0 = w0
        self.rho = rho
        self.phi = phi
        self.omega = omega
        self.max_iter = max_iter
        self.tol = tol

    def __repr__(self):
        return (f"StateDrivenConfig(w0={self.w0}, rho={self.rho}, "
                f"phi={self.phi}, omega={self.omega})")


def coupling_weight(x_i, x_j, cfg):
    """
    W(t) 的核心一行：**状态越差，跨层供给打折越狠**。

        w_ij(t) = w0 / [1 + ρ·(1 − min(x_i, x_j))]

    两边满血 ⇒ w = w0；一边掉到 0 ⇒ w = w0/(1+ρ)。ρ=0 ⇒ 常数耦合（退回 L1）。
    """
    return cfg.w0 / (1.0 + cfg.rho * (1.0 - min(x_i, x_j)))


def _step_states_w(net, cfg, layer, other):
    """算一层节点下一步的供给 s_i(t)，同时把 W(t) 用在跨层项上。"""
    new_x = {}
    for node in net.G[layer].nodes():
        if not net.alive[layer][node]:
            new_x[node] = 0.0
            continue

        degree0 = net.G[layer].degree(node) or 1  # 原始度：分母固定，死邻居贡献 0
        neighbour_supply = sum(
            net.x[layer][j] for j in net.G[layer].neighbors(node) if net.alive[layer][j]
        ) / degree0

        partner = net.partner[layer].get(node)
        partner_alive = partner is not None and net.alive[other][partner]
        if partner_alive:
            x_j = net.x[other][partner]
            cross = coupling_weight(net.x[layer][node], x_j, cfg) * x_j
        else:
            cross = 0.0

        if partner is None:  # 没有伙伴 ⇔ 不吃跨层项（权重不"吞掉"这份供给）
            x = net.x[layer][node]
            new_x[node] = x + cfg.mu * (1.0 - x) + (1.0 - cfg.omega - cfg.mu) * (neighbour_supply - x)
            new_x[node] = min(1.0, max(0.0, new_x[node]))
            continue

        x = net.x[layer][node]
        new_x[node] = (x
                       + cfg.mu * (1.0 - x)
                       + (1.0 - cfg.omega - cfg.mu) * (neighbour_supply - x)
                       + cfg.omega * (cross - x))
        new_x[node] = min(1.0, max(0.0, new_x[node]))
    return new_x


class L2Rule:
    """
    L2 的失效判定：布尔耦合 + W(t) 动力学 + 满足度判据 Φ < φ。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, net):
        newly = list(boolean_coupling_rule(net))

        for _ in range(self.cfg.max_iter):
            new_A = _step_states_w(net, self.cfg, "A", "B")
            new_B = _step_states_w(net, self.cfg, "B", "A")
            diff = 0.0
            for layer, new_x in (("A", new_A), ("B", new_B)):
                diff = max(diff, max(abs(new_x[n] - net.x[layer][n]) for n in new_x))
            net.x["A"], net.x["B"] = new_A, new_B
            if diff < self.cfg.tol:
                break

        for layer in ("A", "B"):
            for node in net.G[layer].nodes():
                if not net.alive[layer][node]:
                    continue
                if net.x[layer][node] < self.cfg.phi:  # Φ_i = x_i / d_i, d_i = 1
                    newly.append((layer, node))
        return newly


def run_experiment(N=300, m=3, n_trials=3, seed=42, save=True):
    """对比：常数耦合（W≡w0，退化档）vs 状态驱动耦合（ρ=1）vs 更严的满意线（φ 更大）。"""
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    fractions = np.linspace(0.0, 0.4, 17)
    settings = {
        "ρ=0, φ=0（退回 L0）": StateDrivenConfig(rho=0.0, phi=0.0),
        "ρ=0, φ=0.8（常数耦合+严 QoS）": StateDrivenConfig(rho=0.0, phi=0.8),
        "ρ=1, φ=0.8（状态驱动）": StateDrivenConfig(rho=1.0, phi=0.8),
        "ρ=3, φ=0.9（强状态驱动）": StateDrivenConfig(rho=3.0, phi=0.9),
    }

    print(f"=== {LEVEL}：耦合随状态退化（供给打折）⇒ 级联更早发生 ===")
    curves = {}
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())
    for label, cfg in settings.items():
        curve = []
        for f in fractions:
            n_remove = int(round(len(nodes_A) * f))
            trials = []
            for _ in range(n_trials):
                net = InterdependentNetwork(G_A, G_B, seed=seed)
                failed = ([] if n_remove == 0
                          else rng.choice(nodes_A, size=n_remove, replace=False))
                trials.append(net.run(failed, rule=L2Rule(cfg))["S_A"])
            curve.append(float(np.mean(trials)))
        curves[label] = curve
        print(f"  {label:<26} → S(f=0.2) = {np.interp(0.2, fractions, curve):.3f}")

    if save:
        plt.figure(figsize=(9, 6))
        for label, curve in curves.items():
            plt.plot(fractions, curve, "o-", markersize=4, label=label)
        plt.xlabel("初始移除比例 f")
        plt.ylabel("存活序参量 S（层 A 的 GCC 占比）")
        plt.title(f"{LEVEL}：耦合强度随状态演化（W(t)）")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        out = RESULTS_DIR / "phase6_L2_state_driven.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图已保存：{out}")

    return {"fractions": list(fractions), "curves": curves}


if __name__ == "__main__":
    run_experiment()
