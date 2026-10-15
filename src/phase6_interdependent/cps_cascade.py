"""
Phase 6 · CPS —— 信息-物理耦合网络主循环（两张耦合张量 W^C / W^S）

【L3 缺了什么】
  L0–L3 是"逐级加物理"的台阶，但每级只用一种机制。真实的信息-物理系统（电网、制造网络）里，
  **物理层和信息层同时在动**，而且层间耦合强度**随状态变化**：
    · 物理层的设备靠信息层下指令 ⇒ 时延越大、设备越退化，指令越使不上劲
    · 信息层靠物理层给状态 ⇒ 设备越退化，传感器看到的信号越糊
  把这两个"打折"写成设计稿 §5.2 的状态函数，就是 W^C（控制依赖）与 W^S（感知依赖）。

【本级的模型】（设计稿 §5.2，符号与设计稿一致）
  每个存活节点：x_i(t+1) = x_i + μ(1−x_i) + λ(邻居供给 − x_i) + ω·W(·)(伙伴状态 − x_i)

    W^C_{iν}(t) = w⁰ · g(τ̂_ν) · κ(h_i)      g(τ̂) = exp(−λ_τ·τ̂)   κ(h) = h^α
        A 层（物理层）用：设备 i 受控制器 ν 的指令 ⇒ 延迟衰减 × 设备自身健康度
    W^S_{νi}(t) = w⁰ · s(h_i) · (1 − q_ν)    s(h) = 1 − β(1−h)
        B 层（信息层）用：传感器 ν 看设备 i 的状态 ⇒ 传感退化 × 感知完整性

  判失效（三个口径，都可单独关掉）：
    · δ 口径（L1）：|1 − x_i/x̃_i| > δ                 ⇒ δ=1 关闭
    · φ 口径（L2）：x_i < φ                           ⇒ φ=0 关闭
    · Φ 口径（本级，设计稿"位置 B"）：Φ_ν = Σ_{存活 i} W^S_{νi} / Σ_i w⁰ < φ_s
                                                      ⇒ φ_s=0 关闭
  每轮顺序：动力学迭代到不动点（**每轮重算 W(t)**）→ 判失效（δ/φ/Φ）→ 布尔耦合 → 超边逻辑 → GCC 剪枝

【⭐退化性检验】
  τ̂=0、q=0、w⁰=1 ⇒ W^C = W^S = 1（与常数耦合完全一样）；
  再 δ=1（关容忍）+ φ=0、φ_s=0 + 超边只用二元 AND ⇒ 必须与 L0 完全一致。

【这一级要学的知识】
  · 张量 vs 标量耦合：W 从常数变成"随 (x_i, τ̂, h, q) 变的表"，这就是 `W(t)` 连续化
  · **新失效通道（设计稿 §5.2 末）**：不需要任何节点死亡，只要 h 下降拉低 s(h)、
    或 τ 上升拉低 g(τ̂)，依赖满足度 Φ 就会跌破 φ_s ⇒ 连续劣化就能让系统崩
  · 为什么最值钱：研究内容 1 的 CSQ1 就在这里 —— W(t) 连续化 + 超边逻辑依赖叠加

运行：
    python src/phase6_interdependent/cps_cascade.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
LAYER_ROLE = {"A": "物理层（受信息层控制 → 用 W^C）",
              "B": "信息层（感知物理层状态 → 用 W^S）"}

__all__ = ["CPSConfig", "CPSNetwork", "CPSRule", "control_weight", "sensing_weight",
           "sensing_satisfaction", "run_cps_dynamics", "healthy_reference",
           "run_experiment", "run_experiment_degradation"]


@dataclass
class CPSConfig:
    """
    主循环的旋钮。μ + λ + ω ≤ 1 才稳定（增量式：自反馈 / 网内 / 跨层）。
       mu    自反馈回弹（无邻居时也不会自己掉到 0 —— L0/L1 踩过的坑）
       lam   网内扩散（邻居供给权重）
       omega 跨层耦合的基础权重（再乘 W^C / W^S）
    —— 设计稿 §5.2 / §6.1 的耦合参数 ——
       w0         静态基线权重 w⁰（工艺给定；退化性检验要求 w⁰=1）
       lambda_tau 时延衰减系数 λ_τ ∈ [0.1, 2]
       alpha      可受控性指数 α ∈ [0.5, 2]（κ(h)=h^α，越退化越不受控）
       beta       感知退化系数 β ∈ [0, 1]（s(h)=1−β(1−h)）
       tau_hat    全局归一化时延 τ̂ ∈ [0, 1]（可被 net.tau_hat 逐节点覆盖）
       q          全局丢包率 q ∈ [0, 1]（可被 net.q 逐节点覆盖）
       target_A   物理层的可获得健康度上限 x⁰_A（=1 全健康；<1 模拟长期退化/持续载荷）
                  —— 没有它就测不出“不删节点、只靠耦合劣化”的新失效通道
    —— 判失效旋钮 ——
       delta  δ 口径：相对偏离容忍带（δ=1 ⇒ 关闭）
       phi    φ 口径：状态绝对底线（φ=0 ⇒ 关闭）
       phi_s  Φ 口径：感知满足度阈值（φ_s=0 ⇒ 关闭）
    """
    mu: float = 0.4
    lam: float = 0.3
    omega: float = 0.3
    w0: float = 1.0
    lambda_tau: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5
    tau_hat: float = 0.0
    q: float = 0.0
    target_A: float = 1.0
    delta: float = 0.2
    phi: float = 0.0
    phi_s: float = 0.0
    tol: float = 1e-6
    max_iter: int = 50

    def __repr__(self):
        return (f"CPSConfig(μ={self.mu}, λ={self.lam}, ω={self.omega}, w⁰={self.w0}, "
                f"λτ={self.lambda_tau}, α={self.alpha}, β={self.beta}, "
                f"τ̂={self.tau_hat}, q={self.q}, x⁰_A={self.target_A}, δ={self.delta}, φ={self.phi}, φs={self.phi_s})")


# ---------------------------------------------------------------- 耦合函数
def control_weight(h_i, tau_hat, w0=1.0, lambda_tau=1.0, alpha=1.0):
    """W^C 的一格。g(τ̂)=exp(−λ_τ τ̂)：指令越滞后越不管用；κ(h)=h^α：设备越退化越不受控。"""
    return w0 * math.exp(-lambda_tau * tau_hat) * (max(float(h_i), 0.0) ** alpha)


def sensing_weight(h_i, q, w0=1.0, beta=0.5):
    """W^S 的一格。s(h)=1−β(1−h)：退化压信号质量；(1−q)：感知完整性。"""
    return w0 * (1.0 - beta * (1.0 - float(h_i))) * (1.0 - q)


def _tau_hat(net, cfg, node):
    return float(getattr(net, "tau_hat", {}).get(node, cfg.tau_hat))


def _q(net, cfg, node):
    return float(getattr(net, "q", {}).get(node, cfg.q))


def _target(net, cfg, layer, node):
    """物理层的健康度天花板：默认全局 cfg.target_A，可用 net.target_A_map 逐设备给（异质退化）。"""
    if layer != "A":
        return 1.0
    return float(getattr(net, "target_A_map", {}).get(node, cfg.target_A))


def _partner_weight(net, cfg, layer, node, x_self, x_partner):
    """按层选张量：A 层（物理）吃 W^C，B 层（信息）吃 W^S。"""
    partner = net.partner[layer].get(node)
    if layer == "A":
        # 设备 node 受控制器 partner 的指令：看对方的时延 + 本设备的健康度
        return control_weight(x_self, _tau_hat(net, cfg, partner) if partner is not None else 0.0,
                              cfg.w0, cfg.lambda_tau, cfg.alpha)
    # 传感器 node 看设备 partner 的状态：看设备的健康度 + 本传感器的丢包率
    return sensing_weight(x_partner, _q(net, cfg, node), cfg.w0, cfg.beta)


def sensing_satisfaction(net, cfg, layer="B"):
    """Φ_ν = Σ_{存活 i} W^S_{νi}(t) / Σ_i w⁰ —— 依赖满足度（设计稿"位置 B"）。

    一对一耦合下每个信息节点只有一个被感知设备，分母就是 w⁰。
    """
    out = {}
    for nu in net.G[layer].nodes():
        if not net.alive[layer][nu]:
            out[nu] = 0.0
            continue
        i = net.partner[layer].get(nu)
        if i is None:                       # 无伙伴 ⇒ 不背这份依赖
            out[nu] = 1.0
            continue
        if not net.alive["A"][i]:
            out[nu] = 0.0
            continue
        out[nu] = sensing_weight(net.x["A"][i], _q(net, cfg, nu), cfg.w0, cfg.beta) / max(cfg.w0, 1e-12)
    return out


# ---------------------------------------------------------------- 动力学
def _step_states(net, cfg, layer, other):
    """一次 Jacobi 更新（层内 + 跨层），返回 {node: x_new}。"""
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
            w = _partner_weight(net, cfg, layer, node, x, x_partner)
            cross = cfg.omega * w * (x_partner - x)
        # 自反馈回弹：物理层可以被“长期退化”压低天花板（x⁰_A < 1），信息层保持 1
        target = (_target(net, cfg, layer, node) if layer == "A" else 1.0)
        value = x + cfg.mu * (target - x) + cfg.lam * (supply - x) + cross
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


# ---------------------------------------------------------------- 主循环规则
class CPSRule:
    """每轮：动力学 → 判失效（δ / φ / Φ 三个口径）→ 布尔耦合 → 超边逻辑。"""

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

        if self.cfg.phi_s > 0.0:            # 设计稿"位置 B"：感知满足度不足 ⇒ 信息节点失效
            for nu, phi in sensing_satisfaction(net, self.cfg).items():
                if net.alive["B"][nu] and phi < self.cfg.phi_s:
                    newly.append(("B", nu))

        newly.extend(boolean_coupling_rule(net))
        for edge in getattr(net, "hyperedges", []):
            if edge.is_active(net.alive):
                continue
            newly.extend(m for m in edge.members if net.alive[m[0]][m[1]])
        return newly


class CPSNetwork(HyperedgeNetwork):
    """信息-物理耦合网络：在 L3 超图网络上，把层间耦合升级成 W^C / W^S 状态函数。

    可选逐节点服务质量：`net.tau_hat = {B 节点: τ̂}`、`net.q = {B 节点: q}`（缺省用 cfg 的全局值）。
    """

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


# ---------------------------------------------------------------- 实验
def run_experiment(N=300, m=3, n_trials=3, seed=42, save=True):
    """
    四个配置逐个拧开旋钮：纯结构 → 加容忍带 δ → 加感知满足度 Φ → 再把状态驱动耦合（λτ/β）打开。
    """
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    units = random_functional_hyperedges(
        HyperedgeNetwork(G_A, G_B, seed=seed), n_edges=60, size=3, rule="k-of-N", k=2, seed=seed)
    fractions = np.linspace(0.0, 0.4, 17)

    settings = {
        "基线：关掉 δ/φ/Φ、耦合常数": CPSConfig(lambda_tau=0.0, beta=0.0, delta=1.0),
        "只加容忍带 δ=0.3": CPSConfig(lambda_tau=0.0, beta=0.0, delta=0.3),
        "只加满足度 Φ=0.75": CPSConfig(lambda_tau=0.0, beta=0.0, delta=1.0, phi_s=0.75),
        "全开：λτ=1 + β=0.5 + δ=0.3 + Φ=0.75": CPSConfig(lambda_tau=1.0, beta=0.5,
                                                          delta=0.3, phi_s=0.75),
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


def run_experiment_degradation(N=300, m=3, healths=None, n_trials=3, seed=42, save=True):
    """
    设计稿 §5.2 的**新失效通道**：不删任何节点，只让设备"慢慢退化"（h 下降）。

    δ 与 φ 全部关掉，只留 Φ 口径 ⇒ 任何失效都只能来自"依赖满足度不足"。
    实测期望：h = 1.0（全健康）时零失效（Φ=1 > φ_s）；h 掉到阈值以下后，
    信息节点先失效、再经布尔耦合把设备拖死 —— **没有初始故障也能级联**。
    """
    healths = np.linspace(1.0, 0.0, 11) if healths is None else np.asarray(healths, dtype=float)
    cfg = CPSConfig(beta=0.5, delta=1.0, phi=0.0, phi_s=0.8)
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    units = random_functional_hyperedges(
        HyperedgeNetwork(G_A, G_B, seed=seed), n_edges=60, size=3, rule="k-of-N", k=2, seed=seed)

    print(f"=== {LEVEL}：只靠耦合劣化的新失效通道（不删节点，δ/φ 关掉，Φ={cfg.phi_s}）===")
    rows, rows_het = [], []
    for h0 in healths:
        for hetero, bucket in ((False, rows), (True, rows_het)):
            Ss, Fs, deads, phis = [], [], [], []
            for t in range(n_trials):
                cfg_h = CPSConfig(beta=cfg.beta, delta=1.0, phi=0.0,
                                  phi_s=cfg.phi_s, target_A=float(h0))
                net = CPSNetwork(G_A, G_B, hyperedges=units, seed=seed)
                if hetero:      # 设备退化程度不同：h_i = h0·U(0.5, 1.0)（换一茬设备就换一次）
                    rng = np.random.default_rng(1000 + t)
                    net.target_A_map = {i: float(min(1.0, h0 * rng.uniform(0.5, 1.0)))
                                        for i in net.G["A"].nodes()}
                net.run([], rule=None)                      # 无人为故障，只做 GCC 对齐
                net.run([], rule=net.cps_rule(cfg_h, None))  # 只靠 x⁰_A 下降驱动
                Ss.append(net.S("A"))
                Fs.append(net.functional_measure())
                deads.append(sum(1 for n in net.G["A"] if not net.alive["A"][n]))
                alive_phi = [v for n, v in sensing_satisfaction(net, cfg_h).items()
                             if net.alive["B"][n]]
                phis.append(min(alive_phi) if alive_phi else 0.0)
            bucket.append((h0, float(np.mean(Ss)), float(np.mean(Fs)),
                           float(np.mean(deads)), float(np.mean(phis))))
        print(f"  设备健康度 h={h0:.2f} → 均匀：Φ_min={rows[-1][4]:.3f} 失效={rows[-1][3]:5.1f} "
              f"S={rows[-1][1]:.3f} F={rows[-1][2]:.3f} | 异质：Φ_min={rows_het[-1][4]:.3f} "
              f"失效={rows_het[-1][3]:5.1f} S={rows_het[-1][1]:.3f} F={rows_het[-1][2]:.3f}")

    arr = np.array(rows)
    arr_het = np.array(rows_het)

    if save:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(arr[:, 0], arr[:, 1], "o-", label="S · 均匀退化")
        ax.plot(arr[:, 0], arr[:, 2], "s-", label="F · 均匀退化")
        ax.plot(arr_het[:, 0], arr_het[:, 1], "o--", alpha=0.7, label="S · 异质退化")
        ax.plot(arr_het[:, 0], arr_het[:, 2], "s--", alpha=0.7, label="F · 异质退化")
        ax.set_xlabel("设备健康度尺度 h（越小越退化）")
        ax.set_ylabel("级联后的序参量")
        ax.set_title(f"只靠耦合劣化的新失效通道（Φ 阈值 {cfg.phi_s}，无初始故障）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(RESULTS_DIR / "phase6_CPS_satisfaction_channel.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图已保存：{RESULTS_DIR / 'phase6_CPS_satisfaction_channel.png'}")
    return {"healths": arr[:, 0], "S": arr[:, 1], "F": arr[:, 2], "failed": arr[:, 3],
            "phi_min": arr[:, 4], "S_het": arr_het[:, 1], "F_het": arr_het[:, 2],
            "failed_het": arr_het[:, 3], "cfg": cfg}


if __name__ == "__main__":
    run_experiment()
    run_experiment_degradation()
