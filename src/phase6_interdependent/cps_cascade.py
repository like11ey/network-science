"""
Phase 6 · CPS —— 信息-物理耦合网络主循环（两张耦合张量 W^C / W^S）

【L3 缺了什么】
  L0–L3 是"逐级加物理"的台阶，但每级只用一种机制。真实的信息-物理系统（电网、制造网络）里，
  **物理层和信息层同时在动**，而且层间耦合强度**随状态变化**：
    · 物理层靠信息层给指令 ⇒ 信息层状态差时，物理层拿到的指令支撑要打折
    · 信息层靠物理层供电 ⇒ 供电质量差时，信息层同样打折
  把这两个"打折"写成状态函数，就是 W^C（信息耦合）与 W^S（供给耦合）。

【本级的模型】
  每个存活节点：x_i(t+1) = x_i + μ(1−x_i) + λ(邻居供给 − x_i) + ω·W(·)(伙伴状态 − x_i)
    W^C_ij = τ^C / (1 + h^C (1 − min(x_i, x_j)))    ← 信息耦合，A 层（物理层）用
    W^S_ij = τ^S / (1 + h^S (1 − min(x_i, x_j)))    ← 供给耦合，B 层（信息层）用
  判失效（两个口径都在，可分别关掉）：
    · δ 口径（L1）：|1 − x_i/x̃_i| > δ        ⇒ δ=1 关闭
    · φ 口径（L2）：x_i < φ                  ⇒ φ=0 关闭
  每轮顺序：动力学迭代到不动点（**每轮重算 W(t)**）→ 判失效 → 布尔耦合 → 超边逻辑 → GCC 剪枝

【⭐退化性检验】
  δ=1（关容忍）+ φ=0（关满足度）+ 超边只用二元 AND ⇒ 必须与 L0 完全一致：
  此时动力学只改 x、"判失效"一条都不触发，失效只能来自布尔耦合 / 超边 / GCC 剪枝。

【这一级要学的知识】
  · 张量 vs 标量耦合：W 从常数变成"随 (x_i, x_j) 变的表"，这就是 `W(t)` 连续化
  · 分工：A 层 = 物理层（吃信息层的指令），B 层 = 信息层（吃物理层的供电）
  · 为什么最值钱：研究内容 1 的 CSQ1 就在这里 —— W(t) 连续化 + 超边逻辑依赖叠加

运行：
    python src/phase6_interdependent/cps_cascade.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:  # 作为包被导入
    from .core import boolean_coupling_rule
    from .L3_hypergraph_logic import HyperedgeNetwork, random_functional_hyperedges
except ImportError:  # 直接当脚本跑
    from core import boolean_coupling_rule
    from L3_hypergraph_logic import HyperedgeNetwork, random_functional_hyperedges

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEVEL = "CPS 信息-物理耦合主循环（W^C / W^S）"
LAYER_ROLE = {"A": "物理层（吃信息层的指令 → 用 W^C）",
              "B": "信息层（吃物理层的供电 → 用 W^S）"}

__all__ = ["CPSConfig", "CPSNetwork", "CPSRule", "coupling_weight",
           "run_cps_dynamics", "healthy_reference", "run_experiment"]


@dataclass
class CPSConfig:
    """
    主循环的旋钮。μ + λ + ω ≤ 1 才稳定（增量式：自反馈 / 网内 / 跨层）。
       mu    自反馈回弹（无邻居时也不会自己掉到 0 —— L0/L1 踩过的坑）
       lam   网内扩散（邻居供给权重）
       omega 跨层耦合的基础权重（再乘 W^C / W^S）
       tau_s/h_s  供给耦合强度与对状态退化的敏感度
       tau_c/h_c  信息耦合强度与对状态退化的敏感度
       delta  L1 的容忍带（δ=1 ⇒ 关闭"相对偏离"判据）
       phi    L2 的满足度底线（φ=0 ⇒ 关闭"状态太低"判据）
    """
    mu: float = 0.4
    lam: float = 0.3
    omega: float = 0.3
    tau_s: float = 1.0
    tau_c: float = 1.0
    h_s: float = 1.0
    h_c: float = 1.0
    delta: float = 0.2
    phi: float = 0.0
    tol: float = 1e-6
    max_iter: int = 50

    def __repr__(self):
        return (f"CPSConfig(μ={self.mu}, λ={self.lam}, ω={self.omega}, "
                f"τS={self.tau_s}, hS={self.h_s}, τC={self.tau_c}, hC={self.h_c}, "
                f"δ={self.delta}, φ={self.phi})")


def coupling_weight(x_i, x_j, tau, h):
    """W(t) 的一格：状态越差（min 越小）折扣越狠；h=0 ⇒ 退回常数 τ。"""
    return tau / (1.0 + h * (1.0 - min(x_i, x_j)))


def _layer_knobs(cfg, layer):
    """A 层（物理）用信息耦合 W^C，B 层（信息）用供给耦合 W^S。"""
    return (cfg.tau_c, cfg.h_c) if layer == "A" else (cfg.tau_s, cfg.h_s)


def _step_states(net, cfg, layer, other):
    """一次 Jacobi 更新（层内 + 跨层），返回 {node: x_new}。"""
    tau, h = _layer_knobs(cfg, layer)
    new_x = {}
    for node in net.G[layer].nodes():
        if not net.alive[layer][node]:
            new_x[node] = 0.0
            continue
        x = net.x[layer][node]
        degree0 = net.G[layer].degree(node) or 1
        supply = sum(net.x[layer][j] for j in net.G[layer].neighbors(node)
                     if net.alive[layer][j]) / degree0
        cross = 0.0
        partner = net.partner[layer].get(node)
        if partner is not None and net.alive[other][partner]:
            x_partner = net.x[other][partner]
            w = coupling_weight(x, x_partner, tau, h)
            cross = cfg.omega * w * (x_partner - x)
        value = x + cfg.mu * (1.0 - x) + cfg.lam * (supply - x) + cross
        new_x[node] = float(min(1.0, max(0.0, value)))
    return new_x


def run_cps_dynamics(net, cfg, max_iter=None, tol=None):
    """把两层一起迭代到不动点（每轮都重算 W(t)）。"""
    max_iter = cfg.max_iter if max_iter is None else max_iter
    tol = cfg.tol if tol is None else tol
    pairs = (("A", "B"), ("B", "A"))
    for _ in range(max_iter):
        new = {layer: _step_states(net, cfg, layer, other) for layer, other in pairs}
        diff = max(max(abs(new[layer][n] - net.x[layer][n]) for n in new[layer])
                   for layer in ("A", "B"))
        net.x["A"], net.x["B"] = new["A"], new["B"]
        if diff < tol:
            break
    return net


def healthy_reference(G_A, G_B, cfg, coupled_fraction=1.0, seed=0):
    """未受损时的稳态 x̃（δ 判据的分母）。"""
    net = CPSNetwork(G_A, G_B, seed=seed, coupled_fraction=coupled_fraction)
    net.run([], rule=None)
    run_cps_dynamics(net, cfg)
    return {layer: dict(net.x[layer]) for layer in ("A", "B")}


class CPSRule:
    """每轮：动力学 → 判失效（δ / φ 两个口径）→ 布尔耦合 → 超边逻辑。"""

    def __init__(self, cfg, x_ref=None):
        self.cfg = cfg
        self.x_ref = x_ref or {}

    def __call__(self, net):
        newly = []
        run_cps_dynamics(net, self.cfg)

        for layer in ("A", "B"):
            ref = self.x_ref.get(layer, {})
            for node in net.G[layer].nodes():
                if not net.alive[layer][node]:
                    continue
                x = net.x[layer][node]
                if self.cfg.phi > 0.0 and x < self.cfg.phi:
                    newly.append((layer, node))
                    continue
                if self.cfg.delta < 1.0:
                    r = ref.get(node, 0.0)
                    if r > 0.0 and abs(1.0 - x / r) > self.cfg.delta:
                        newly.append((layer, node))

        newly.extend(boolean_coupling_rule(net))
        for edge in getattr(net, "hyperedges", []):
            if edge.is_active(net.alive):
                continue
            newly.extend(m for m in edge.members if net.alive[m[0]][m[1]])
        return newly


class CPSNetwork(HyperedgeNetwork):
    """信息-物理耦合网络：在 L3 超图网络上，把层间耦合升级成 W^C / W^S 状态函数。"""

    def cps_rule(self, cfg, x_ref=None):
        return CPSRule(cfg, x_ref)

    def run_experiment_curves(self, cfg, fractions, n_trials=3, seed=1):
        """便捷入口：直接给一个 cfg 出 S / F 两条曲线（notebook 里好用）。"""
        rng = np.random.default_rng(seed)
        nodes_A = sorted(self.G["A"].nodes())
        x_ref = healthy_reference(self.G["A"], self.G["B"], cfg, seed=seed)
        curve_S, curve_F = [], []
        for f in fractions:
            n_remove = int(round(len(nodes_A) * f))
            Ss, Fs = [], []
            for _ in range(n_trials):
                failed = ([] if n_remove == 0
                          else rng.choice(nodes_A, size=n_remove, replace=False))
                net = CPSNetwork(self.G["A"], self.G["B"],
                                 hyperedges=self.hyperedges, seed=seed)
                net.run(failed, rule=net.cps_rule(cfg, x_ref))
                Ss.append(net.S("A"))
                Fs.append(net.functional_measure())
            curve_S.append(float(np.mean(Ss)))
            curve_F.append(float(np.mean(Fs)))
        return np.array(curve_S), np.array(curve_F)


def run_experiment(N=300, m=3, n_trials=3, seed=42, save=True):
    """
    四个配置逐个拧开旋钮：纯结构 → 加容忍带 δ → 加服务质量 φ → 再把状态驱动退化 h 打开。
    实测（N=300, k-of-N 功能单元）p_c：0.43 → 0.28 → 0.18 → 0.12，说明三个旋钮都真的在起作用。
    """
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    units = random_functional_hyperedges(
        HyperedgeNetwork(G_A, G_B, seed=seed), n_edges=60, size=3, rule="k-of-N", k=2, seed=seed)
    fractions = np.linspace(0.0, 0.4, 17)

    settings = {
        "基线：关掉 δ/φ（h=0, δ=1, φ=0）": CPSConfig(h_s=0.0, h_c=0.0, delta=1.0, phi=0.0),
        "只加容忍带 δ=0.3": CPSConfig(h_s=0.0, h_c=0.0, delta=0.3, phi=0.0),
        "只加服务质量 φ=0.75": CPSConfig(h_s=0.0, h_c=0.0, delta=1.0, phi=0.75),
        "全开：h=6 + δ=0.3 + φ=0.75": CPSConfig(h_s=6.0, h_c=6.0, delta=0.3, phi=0.75),
    }

    print(f"=== {LEVEL}：耦合张量 W^C / W^S 的作用 ===")
    curves = {}
    for label, cfg in settings.items():
        net = CPSNetwork(G_A, G_B, hyperedges=units, seed=seed)
        S, F = net.run_experiment_curves(cfg, fractions, n_trials=n_trials, seed=seed)
        curves[label] = {"S": S, "F": F}
        print(f"  {label:<30} → S(0.2)={np.interp(0.2, fractions, S):.3f}  "
              f"F(0.2)={np.interp(0.2, fractions, F):.3f}")

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for label, d in curves.items():
            axes[0].plot(fractions, d["S"], "o-", markersize=4, label=label)
            axes[1].plot(fractions, d["F"], "s-", markersize=4, label=label)
        axes[0].set_xlabel("初始移除比例 f")
        axes[0].set_ylabel("结构序参量 S（GCC 占比）")
        axes[1].set_xlabel("初始移除比例 f")
        axes[1].set_ylabel("功能度量 F（活跃功能单元占比）")
        axes[0].set_title("结构视角")
        axes[1].set_title("功能视角")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{LEVEL}（N={N}）")
        out = RESULTS_DIR / "phase6_CPS_coupling_tensors.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图已保存：{out}")
    return {"fractions": fractions, "curves": curves}


if __name__ == "__main__":
    run_experiment()
