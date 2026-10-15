"""
Phase 6 · L1 —— 加网内自动力学（自动力学 M(x) + δ 容忍判据 + 跨层权重 ω）

【L0 缺了什么】
  L0 里节点只有"活/死"，一个节点只要还连着网络就一切正常。
  现实中不是这样：**故障是先"性能下降"，降到超出容忍范围才算失效**。
  这正是 PNAS 2019 相对 Buldyrev 2010 的关键进步：把 0/1 生死，
  换成连续状态 x ∈ [0,1] 加一个容忍参数 δ。

【本级的模型（本项目口径，汇报时要说明）】
  1. 每个节点有一个功能量 x_i ∈ [0,1]（1 = 正常，0 = 完全失效）。
  2. 自动力学（每一步都算一遍，直到稳定）：
         x_i(t+1) = x_i(t)
                  + μ·(1 − x_i(t))                        ← 自反馈：往健康水平回弹（论文的 W(x_i)）
                  + λ·( Σ_{存活 j} x_j(t) / k_i⁰ − x_i(t) )  ← 网内供给（论文的 Q 项）
                  + ω·( x_partner(t) − x_i(t) )             ← 跨层供给（无伙伴的节点不吃这一项）
     各项含义：
         μ       —— 自反馈强度：没有外援时我自己能回到多少（μ=0 就是"纯被邻居供养"）
         λ       —— 网内依赖强度（邻居给我供多少）
         ω       —— 跨层依赖强度（另一层的伙伴给我供多少）
         k_i⁰    —— 节点 i 的**原始度**（分母固定，死掉的邻居就是实实在在少一份供给）
     **完整网络下这个动力学的不动点是 x = 1（人人满血）**，这就是健康参考值 x̃_i。

     ⚠️ 为什么必须有 μ（2026-10-15 修，踩坑记录）：
       早期版本写成 x_i ← (1−ω)[(1−λ)x_i + λ·⟨x_j⟩_存活邻居] + ω·x_partner，
       并且邻居项用"**存活邻居的平均**"。两个毛病叠加 ⇒ 缺口被抹平后又沿全网扩散：
       实测只删 5% 的节点，收敛后 **285/285 个存活节点全部低于 0.8、均值 0.574**，
       于是任何有限 δ 都判"全体超差"，整网在 f≈0.03 就崩，δ 完全失去旋钮意义。
       根因：论文 Eq.1 的 dx_i/dt = W(x_i) + Σ_j A_ij Q(x_i,x_j) 里 W(x_i) 是**自反馈**
       （节点自身的产生/衰减），它给系统提供回弹力；用平均又把死邻居的缺口归一掉了。
       修法：① 加自反馈项 μ·(1−x_i)；② 邻居项按**原始度**归一（死邻居贡献 0，不被归一化补偿）。
  3. 容忍判据（本级的核心）：
         |1 − x_i / x̃_i| > δ  →  判定该节点失效
     通俗说法：性能掉出 δ 的容忍带就"停机"，而不是断开才停机。
     论文口径（Duan 2019）：δ 是"调光旋钮"——δ 小 = 挑剔易失效；δ 足够大 ⇒ 退回纯结构依赖。
     论文取值：Fig 3B 用 δ=1 vs 0.15；Fig 4D 用 δ=0.05/0.08/0.10/0.15。
  4. 仍有布尔耦合（伙伴死了我也死）+ GCC 剪枝兜底。

【⭐ 退化性检验（科研纪律）】
  令 δ = 1（容忍 100%，性能再掉也不停机）⇒ 本级必须**完全退回 L0**。
  这不是"顺便测一下"，而是**模型可信度的证据**：加的东西在极端参数下不改变老结论。
  见 tests/test_interdependent.py::test_L1_degenerates_to_L0。

【这一级要学的知识】
  · 不动点与迭代收敛：为什么反复算会稳定下来
  · 状态量 vs 生死量：连续模型怎么"包含"离散模型
  · 线性耦合算子：邻居平均 = 图拉普拉斯的一阶形式（一句话知道就行）
  · 容忍阈值 δ 的作用：δ 小 = 敏感 = 更容易级联

运行：
    python src/phase6_interdependent/L1_intra_layer_dynamics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:  # 作为包被导入（pytest / python -m）
    from .core import InterdependentNetwork, boolean_coupling_rule
except ImportError:  # 直接当脚本跑
    from core import InterdependentNetwork, boolean_coupling_rule

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEVEL = "L1 网内自动力学"
__all__ = ["AutoDynamicsConfig", "healthy_reference", "run_dynamics", "L1Rule", "run_experiment"]


class AutoDynamicsConfig:
    """
    自动力学的四个旋钮（L1 的全部可调参数）。

    mu   (μ): 自反馈强度（回弹到健康水平 1），对应论文 Eq.1 的 W(x_i)
    lam  (λ): 网内依赖强度，x 越靠邻居供给
    omega(ω): 跨层依赖强度，x 越靠另一层的伙伴
    delta(δ): 容忍带宽，性能掉出 δ 才停机（δ=1 ⇒ 永不因性能停机 ⇒ 退回 L0）

    ⚠️ μ+λ+ω ≤ 1 才有稳定不动点（默认 0.4+0.3+0.3 = 1.0）。
    """

    def __init__(self, mu=0.4, lam=0.3, omega=0.3, delta=0.2, tol=1e-6, max_iter=100):
        self.mu = mu
        self.lam = lam
        self.omega = omega
        self.delta = delta
        self.tol = tol
        self.max_iter = max_iter

    def __repr__(self):
        return (f"AutoDynamicsConfig(mu={self.mu}, lam={self.lam}, omega={self.omega}, "
                f"delta={self.delta})")


def _step_states(net, cfg, layer, other):
    """
    算一层节点的下一步状态 x(t+1)。

    只考虑**当前存活**的邻居（失效邻居对 x 的贡献为 0，相当于断路）。

    ⚠️ 邻居项按**原始度 k⁰ 归一**，不是按存活邻居个数归一：
       按存活个数归一 = "少一个邻居，剩下的邻居把均值拉回 1" ⇒ 缺口被抹平；
       按原始度归一 = "死一个邻居就是少 1/k⁰ 份供给" ⇒ 缺口真实存在（论文的 invisible path）。
    """
    new_x = {}
    for node in net.G[layer].nodes():
        if not net.alive[layer][node]:
            new_x[node] = 0.0
            continue

        degree0 = net.G[layer].degree(node) or 1  # 原始度：分母固定
        neighbour_supply = sum(
            net.x[layer][j] for j in net.G[layer].neighbors(node) if net.alive[layer][j]
        ) / degree0

        partner = net.partner[layer].get(node)

        x = net.x[layer][node]
        # 自反馈回弹（论文的 W(x_i)）+ 网内供给（论文的 Q 项）
        x_new = x + cfg.mu * (1.0 - x) + cfg.lam * (neighbour_supply - x)

        # 跨层供给：**只有真的有伙伴的节点才吃这一项**
        # （coupled_fraction<1 时部分节点没有伙伴；若给它们按 0 伙伴算，
        #   就会凭空多背一份 ω 欠供，健康网自己就先不合格——2026-10-15 修）
        if partner is not None:
            partner_x = net.x[other][partner] if net.alive[other][partner] else 0.0
            x_new += cfg.omega * (partner_x - x)

        new_x[node] = min(1.0, max(0.0, x_new))  # 功能量夹在 [0,1]
    return new_x


def run_dynamics(net, cfg, max_iter=None, tol=None):
    """
    在当前存活结构上把状态 x 迭代到不动点（或到最大步数）。

    学习点：**收敛判据**。每次迭代对比前后状态的最大变化量，
    小于 tol 就认为"不动了"，不再浪费算力。工程代码里到处是这个套路。
    """
    max_iter = cfg.max_iter if max_iter is None else max_iter
    tol = cfg.tol if tol is None else tol
    for _ in range(max_iter):
        new_A = _step_states(net, cfg, "A", "B")
        new_B = _step_states(net, cfg, "B", "A")
        diff = max(
            max(abs(new_A[n] - net.x["A"][n]) for n in new_A) if new_A else 0.0,
            max(abs(new_B[n] - net.x["B"][n]) for n in new_B) if new_B else 0.0,
        )
        net.x["A"], net.x["B"] = new_A, new_B
        if diff < tol:
            break
    return net.x


def healthy_reference(G_A, G_B, cfg, coupled_fraction=1.0, seed=0):
    """
    量健康参考值 x̃_i：在**完好网络**上跑一遍自动力学。

    为什么要参考值？因为容忍判据是"相对自己正常水平掉了多少"，
    所以必须先知道"正常水平"是多少。完好网络上的不动点是 x=1（人人满血），
    但代码写成通用形式，将来换成非线性 M(x) 时不用改。
    """
    net = InterdependentNetwork(G_A, G_B, coupled_fraction=coupled_fraction, seed=seed)
    net.run([], rule=None)  # 空初始故障 = 完好网络
    run_dynamics(net, cfg)
    return {layer: dict(net.x[layer]) for layer in ("A", "B")}


class L1Rule:
    """
    把 L1 的失效判定包成 core.run() 认识的 rule 函数。

    一轮里做三件事：
      1. 布尔耦合：伙伴死了我也死（L0 的规则照旧）
      2. 跑自动力学到稳定
      3. 容忍判据：|1 − x_i/x̃_i| > δ 的节点判失效
    """

    def __init__(self, cfg, x_ref):
        self.cfg = cfg
        self.x_ref = x_ref

    def __call__(self, net):
        newly = list(boolean_coupling_rule(net))
        run_dynamics(net, self.cfg)

        for layer in ("A", "B"):
            for node in net.G[layer].nodes():
                if not net.alive[layer][node]:
                    continue
                ref = self.x_ref[layer][node]
                if ref <= 0:
                    continue
                if abs(1.0 - net.x[layer][node] / ref) > self.cfg.delta:
                    newly.append((layer, node))
        return newly


def run_experiment(N=300, m=3, n_trials=3, seed=42, save=True):
    """扫容忍带宽 δ，看它对级联后果的影响（δ 小 ⇒ 更脆弱）。"""
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    fractions = np.linspace(0.0, 0.4, 17)
    deltas = [0.05, 0.1, 0.15, 0.3, 1.0]  # 前四档对齐 Duan 2019 Fig.4D，1.0 = 退化档

    print(f"=== {LEVEL}：容忍带宽 δ 越小越脆弱（μ=0.4, λ=0.3, ω=0.3）===")
    print("  δ=1.0 这一行应当与 L0 完全一致（退化性检验）")

    curves = {}
    for delta in deltas:
        cfg = AutoDynamicsConfig(mu=0.4, lam=0.3, omega=0.3, delta=delta)
        x_ref = healthy_reference(G_A, G_B, cfg)
        rng = np.random.default_rng(seed)
        nodes_A = sorted(G_A.nodes())
        curve = []
        for f in fractions:
            n_remove = int(round(len(nodes_A) * f))
            trials = []
            for _ in range(n_trials):
                net = InterdependentNetwork(G_A, G_B, seed=seed)
                failed = ([] if n_remove == 0
                          else rng.choice(nodes_A, size=n_remove, replace=False))
                trials.append(net.run(failed, rule=L1Rule(cfg, x_ref))["S_A"])
            curve.append(float(np.mean(trials)))
        curves[delta] = curve
        print(f"  δ={delta:<4} → S(f=0.2)={np.interp(0.2, fractions, curve):.3f}")

    if save:
        plt.figure(figsize=(9, 6))
        for delta, curve in curves.items():
            style = "--" if delta == 1.0 else "-"
            plt.plot(fractions, curve, "o" + style, markersize=4,
                     label=f"δ = {delta}" + ("（= L0 退化）" if delta == 1.0 else ""))
        plt.xlabel("初始移除比例 f")
        plt.ylabel("存活序参量 S（层 A 的 GCC 占比）")
        plt.title(f"{LEVEL}：性能降级到停机（δ 容忍带）")
        plt.legend()
        plt.grid(alpha=0.3)
        out = RESULTS_DIR / "phase6_L1_tolerance.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图已保存：{out}")

    return {"fractions": list(fractions), "curves": curves}


if __name__ == "__main__":
    run_experiment()
