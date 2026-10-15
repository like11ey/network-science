"""
Phase 6 · L3 —— 加超图逻辑依赖（超边 + AND/OR/k-of-N 激活 + 功能度量 F(t)）

【L2 缺了什么】
  L1/L2 的依赖都是**成对**的：A 的第 i 个节点依赖 B 的第 i 个节点。
  但真实系统里"干活"的单元往往**由一组节点共同支撑**，而且是有逻辑的：
    · 一条装配线要 3 个零件同时到位（**AND**，全活才工作）
    · 供电可以由任一条备用线路顶上（**OR**，任意一个活着就还能工作）
    · 一条生产线的 5 台设备里坏 1 台还能降级运行（**k-of-N**，至少 k 台活着）
  这种"多个节点一起决定一个功能单元是否可用"的关系就是**超边**，
  整张图叫**超图**——这也是导师点名"超图建模是理论基石"的落点。

【本级的模型】
  1. 超边（功能单元）：`Hyperedge(members, rule, k)`，members = [(层, 节点), ...]
       rule = 'AND'    : 全部成员存活 ⇒ 活跃
       rule = 'OR'     : 任一成员存活 ⇒ 活跃
       rule = 'k-of-N' : 至少 k 个成员存活 ⇒ 活跃
  2. 每轮判定顺序（和 L1/L2 一致，便于对拍）：
       ① 布尔耦合照旧（伙伴死了我也死）
       ② 算每条超边的活跃性
       ③ **不属于任何活跃超边的节点 ⇒ 失去功能**（没有功能单元还需要它）
       ④ GCC 剪枝兜底
  3. 功能度量 F(t) = 活跃超边数 / 超边总数 —— 比"存活节点比例"更贴近业务：
     电网里叫"供电覆盖率"，制造里叫"产能达成率"。
  4. 退化性检验：只用**二元 AND 超边**（等价成对耦合）⇒ 必须与 L0/L1/L2 的
     成对模型一致。见 tests/test_interdependent.py::test_L3_pairs_degenerate_to_L0。

【这一级要学的知识】
  · 超图 vs 普通图：一条边可以连多个节点（networkx 用 HyperGraph 或在代码里自己定义）
  · 逻辑门语义：AND / OR / k-of-N 是"故障树（fault tree）"的基本构件
  · 功能度量 F(t)：从"结构还连着"升级到"业务还在跑"
  · 为什么这一步最值钱：L0–L2 文献里有现成做法，**L3 的高阶逻辑依赖是本项目要做的活**

运行：
    python src/phase6_interdependent/L3_hypergraph_logic.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

LEVEL = "L3 超图逻辑依赖"
__all__ = [
    "Hyperedge", "HyperedgeNetwork", "pair_hyperedges", "random_functional_hyperedges",
    "functional_measure", "run_experiment",
]


@dataclass
class Hyperedge:
    """
    一个功能单元（超边）。

    members : [(层, 节点), ...]，成员可以跨层、也可以同一层
    rule    : 'AND' / 'OR' / 'k-of-N'
    k       : 仅 k-of-N 用，至少 k 个成员存活才活跃
    name    : 便于打印的名字，如 "装配线-1"
    """

    members: list
    rule: str = "AND"
    k: int = 1
    name: str = ""

    def is_active(self, alive):
        """按规则判断这条超边是否还在工作。"""
        alive_count = sum(1 for layer, node in self.members if alive[layer][node])
        if self.rule == "AND":
            return alive_count == len(self.members)
        if self.rule == "OR":
            return alive_count >= 1
        if self.rule == "k-of-N":
            return alive_count >= self.k
        raise ValueError(f"不认识的激活规则: {self.rule}")


class HyperedgeNetwork(InterdependentNetwork):
    """
    带超边的双层网络：在 core.InterdependentNetwork 上多挂一组功能单元。

    继承（inheritance）的用法在这里很典型：**父类负责记账，子类负责新物理**。
    级联循环直接复用父类的 run()，只把 rule 换成超图判定。
    """

    def __init__(self, G_A, G_B, hyperedges=None, coupling="one_to_one",
                 coupled_fraction=1.0, seed=0):
        super().__init__(G_A, G_B, coupling=coupling,
                         coupled_fraction=coupled_fraction, seed=seed)
        self.hyperedges = list(hyperedges or [])
        if not self.hyperedges:  # 默认：层间耦合的二元 AND 版本
            self.hyperedges = pair_hyperedges(self)

    def active_flags(self):
        """每条超边当前是否活跃（列表，顺序与 self.hyperedges 一致）。"""
        return [e.is_active(self.alive) for e in self.hyperedges]

    def functional_measure(self):
        """功能度量 F = 活跃超边 / 全部超边。没有超边时按 1 处理。"""
        if not self.hyperedges:
            return 1.0
        return float(np.mean(self.active_flags()))

    def hyperedge_rule(self, extra_rule=None):
        """
        生成 core.run() 用的 rule：布尔耦合 + 超边逻辑 + （可选的）额外动力学判定。
        """
        def _rule(net):
            newly = list(boolean_coupling_rule(net))
            if extra_rule is not None:
                newly.extend(extra_rule(net))
            for edge in net.hyperedges:
                if edge.is_active(net.alive):
                    continue
                # 超边失效 ⇒ 成员失去功能（没有功能单元还需要它们）
                newly.extend(m for m in edge.members if net.alive[m[0]][m[1]])
            return newly

        return _rule


def pair_hyperedges(net, layer_pairs=(("A", "B"),)):
    """
    把"层间成对耦合"写成二元 AND 超边 —— L3 用来做退化性检验的基线。

    AND 语义：这条边（依赖关系）只有在**两边都活着**时才成立；
    任一边死了，另一边就"不再被这条边需要"⇒ 失效。这与 L0 的布尔耦合完全等价。
    """
    edges = []
    for layer_a, layer_b in layer_pairs:
        for a, b in net.partner[layer_a].items():
            edges.append(Hyperedge(members=[(layer_a, a), (layer_b, b)],
                                   rule="AND", name=f"pair:{a}-{b}"))
    return edges


def random_functional_hyperedges(net, n_edges=40, size=3, rule="AND", k=None,
                                 layers=("A",), seed=0):
    """
    随机生成"功能单元"超边：每个单元由同层（或跨层）的 size 个节点共同支撑。

    这是"制造网络里的装配单元/电网里的供电片区"的最小抽象。
    k-of-N 规则下，k 默认取 ⌈size/2⌉（坏一半才停机，即降级运行）。
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n_edges):
        layer = layers[i % len(layers)]
        nodes = sorted(net.G[layer].nodes())
        if len(nodes) < size:
            continue
        members = [(layer, n) for n in rng.choice(nodes, size=size, replace=False)]
        edges.append(Hyperedge(members=members, rule=rule,
                               k=(k if k is not None else int(np.ceil(size / 2))),
                               name=f"{rule}-unit-{i}"))
    return edges


def functional_measure(net):
    """对外的快捷函数：直接取一张网的 F 值。"""
    return net.functional_measure()


def run_experiment(N=300, m=3, n_trials=3, seed=42, save=True):
    """
    对比四种功能单元语义在同样攻击下的后果：
        AND（全活才行）→ 最脆弱
        k-of-N（至少一半活着）
        OR（任一活着就行）→ 最鲁棒
    """
    G_A = nx.barabasi_albert_graph(N, m, seed=seed)
    G_B = nx.barabasi_albert_graph(N, m, seed=seed + 1)
    fractions = np.linspace(0.0, 0.4, 17)
    # 固定一套功能单元（用 seed 固定，保证不同规则对比的是同一批单元）
    template = HyperedgeNetwork(G_A, G_B, hyperedges=[], seed=seed)
    units = random_functional_hyperedges(template, n_edges=60, size=3,
                                         rule="AND", seed=seed)

    settings = {
        "AND（全活才工作）": [Hyperedge(list(e.members), "AND", name=e.name) for e in units],
        "k-of-N（≥2 台活着）": [Hyperedge(list(e.members), "k-of-N", k=2, name=e.name) for e in units],
        "OR（任一活着就工作）": [Hyperedge(list(e.members), "OR", name=e.name) for e in units],
    }

    print(f"=== {LEVEL}：功能单元的逻辑语义决定韧性 ===")
    curves = {}
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())
    for label, edges in settings.items():
        curve_S, curve_F = [], []
        for f in fractions:
            n_remove = int(round(len(nodes_A) * f))
            trials_S, trials_F = [], []
            for _ in range(n_trials):
                failed = ([] if n_remove == 0
                          else rng.choice(nodes_A, size=n_remove, replace=False))
                net = HyperedgeNetwork(G_A, G_B, hyperedges=edges, seed=seed)
                net.run(failed, rule=net.hyperedge_rule())
                trials_S.append(net.S("A"))
                trials_F.append(net.functional_measure())
            curve_S.append(float(np.mean(trials_S)))
            curve_F.append(float(np.mean(trials_F)))
        curves[label] = {"S": curve_S, "F": curve_F}
        print(f"  {label:<22} → S(f=0.2)={np.interp(0.2, fractions, curve_S):.3f}  "
              f"F(f=0.2)={np.interp(0.2, fractions, curve_F):.3f}")

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
        axes[1].set_title("功能视角（超图逻辑）")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        fig.suptitle(f"{LEVEL}：AND / k-of-N / OR 的韧性差异")
        out = RESULTS_DIR / "phase6_L3_hypergraph_logic.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  图已保存：{out}")

    return {"fractions": list(fractions), "curves": curves}


if __name__ == "__main__":
    run_experiment()
