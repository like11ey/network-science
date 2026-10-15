"""
Phase 6 核心 —— 双层相依网络容器 + 级联迭代骨架

【这个文件在整个项目里的位置】
  研究内容 1「动态耦合超图建模」要分四级台阶做（L0→L3），四级共用一个骨架：

      L0 静态结构依赖     : core.py 的布尔耦合 + GCC 剪枝（本文件）
      L1 + 网内自动力学    : L1_intra_layer_dynamics.py
      L2 + 状态驱动耦合 W(t): L2_state_driven_coupling.py
      L3 + 超图逻辑依赖    : L3_hypergraph_logic.py

  骨架负责"记账"：谁活着、谁跟谁耦合、与主干断开该算失效；
  每一级只负责回答一个问题：**"这一轮，哪些节点会失去功能？"**（即 rule）

【学科基础知识（边读代码边学）】
  1. 连通分量 / 巨连通分量（GCC）
     一张图里互相连得通的节点构成一个"连通分量"；节点最多的那个叫巨连通分量。
     网络科学里常用"节点是否还在 GCC 里"当作"它还能不能正常工作"——
     孤立的变电站、断网的终端，即使物理完好也算失去功能。
  2. 不动点迭代（fixed-point iteration）
     级联 = 反复做同一件事直到不再有新变化：
         判失效 → 更新状态 → 再判失效 → …… 直到一轮下来没人新失效
     这个"不再变化"的终态就是不动点。所有级联模型本质都是求不动点。
  3. 依赖注入（dependency injection）
     把"失效规则"当参数传进 run()，骨架代码就与具体物理机制解耦，
     加一个新台阶 = 写一个新的 rule 函数，不用改骨架。
  4. 一阶相变 vs 二阶相变
     随移除比例 f 增大，存活比例 S 若是**连续**下降 = 二阶（单层网络）；
     若是**突然跳落**（一处小扰动导致全网崩塌）= 一阶。这是 PNAS 2019 的核心现象。
"""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = ["InterdependentNetwork", "boolean_coupling_rule", "build_coupling"]


def build_coupling(nodes_A, nodes_B, coupled_fraction=1.0, mode="one_to_one", seed=0):
    """
    构造层间耦合关系（A 里的节点 ↔ B 里的节点）。

    参数:
        nodes_A, nodes_B : 两层的节点列表
        coupled_fraction : 耦合比例 q ∈ [0, 1]。q = 1 表示全部耦合（强耦合）；
                           q 小表示只有少数节点互相依赖（弱耦合）——
                           这正是 PNAS 2019 图 3A 里"耦合强度决定相变阶数"的那个旋钮。
        mode             : 'one_to_one'（一一对应，本项目的默认口径）
                           'random'（随机配对，同样 q 个节点）
        seed             : 随机种子（固定种子 = 结果可复现，科研代码的硬要求）

    返回:
        dict: {'A': {a: b, ...}, 'B': {b: a, ...}} —— 双向字典，两个方向都能查
    """
    nodes_A = sorted(nodes_A)
    nodes_B = sorted(nodes_B)
    rng = np.random.default_rng(seed)

    n_couple = int(round(min(len(nodes_A), len(nodes_B)) * coupled_fraction))
    n_couple = max(0, min(n_couple, min(len(nodes_A), len(nodes_B))))
    if n_couple == 0:
        return {"A": {}, "B": {}}

    picked_A = nodes_A[:n_couple]           # one_to_one：按编号顺序取
    picked_B = list(nodes_B)
    if mode == "random":
        rng.shuffle(picked_B)
    picked_B = picked_B[:n_couple]

    forward = {a: b for a, b in zip(picked_A, picked_B)}
    backward = {b: a for a, b in forward.items()}
    return {"A": forward, "B": backward}


class InterdependentNetwork:
    """
    双层相依网络的"记账本"。

    两层网络的物理含义由使用者决定，例如：
        层 A = 电力网（物理层），层 B = 通信网（信息层）
    耦合关系表示"谁依赖谁"：A 的节点 a 依赖 B 的节点 partner[a]。
    """

    def __init__(self, G_A, G_B, coupling="one_to_one", coupled_fraction=1.0, seed=0):
        self.G = {"A": G_A.copy(), "B": G_B.copy()}
        self.coupled_fraction = coupled_fraction
        self.partner = build_coupling(
            self.G["A"].nodes(), self.G["B"].nodes(),
            coupled_fraction=coupled_fraction, mode=coupling, seed=seed,
        )
        self.reset()

    # ---------- 状态管理 ----------

    def reset(self, x0=1.0):
        """
        复位：所有节点存活，功能量 x 全部回到 x0。

        x（功能量/状态量）是 L1 之后才用到的概念：0 = 完全失去功能，1 = 满负荷正常。
        L0 只看存活/失效，不看 x；L1 起用 x 表达"虽然没被拆掉，但功能已经跌出容忍范围"。
        """
        self.alive = {
            layer: {n: True for n in self.G[layer].nodes()} for layer in ("A", "B")
        }
        self.x = {
            layer: {n: float(x0) for n in self.G[layer].nodes()} for layer in ("A", "B")
        }
        self.steps = 0
        self.history = []

    def n_nodes(self, layer):
        return self.G[layer].number_of_nodes()

    def is_alive(self, layer, node):
        return self.alive[layer][node]

    def kill(self, layer, node):
        """标记节点失效（功能量同时归零）。"""
        self.alive[layer][node] = False
        self.x[layer][node] = 0.0

    def alive_nodes(self, layer):
        return [n for n in self.G[layer].nodes() if self.alive[layer][n]]

    # ---------- 结构功能：GCC 剪枝 ----------

    def gcc(self, layer):
        """
        返回该层**存活节点**子图里的巨连通分量（节点集合）。

        学习点：networkx.connected_components 返回的是"分量的迭代器"，
        用 max(..., key=len) 挑出最大的那个。空图要单独处理，否则报错。
        """
        nodes = self.alive_nodes(layer)
        if not nodes:
            return set()
        sub = self.G[layer].subgraph(nodes)
        if sub.number_of_nodes() == 0:
            return set()
        return set(max(nx.connected_components(sub), key=len))

    def prune_gcc(self, layer):
        """把"还在活着、但已被挤出巨连通分量"的节点判为失效，返回新失效集合。"""
        gcc = self.gcc(layer)
        newly = set()
        for n in self.alive_nodes(layer):
            if n not in gcc:
                self.kill(layer, n)
                newly.add(n)
        return newly

    def order_parameter(self, layer="A"):
        """
        序参量 S：GCC 里的节点数 / 该层总节点数。

        这是相变实验里画在 y 轴上的那个量：S 从 1 掉到 0 的过程就是网络崩溃的过程。
        """
        return len(self.gcc(layer)) / max(1, self.n_nodes(layer))

    def S(self, layer="A"):
        """order_parameter 的简写（论文里习惯记 S）。"""
        return self.order_parameter(layer)

    # ---------- 级联骨架 ----------

    def run(self, initial_failures, rule=None, max_steps=200, record=False):
        """
        跑一次级联，直到不动点。

        参数:
            initial_failures : 初始故障，可写成 {层: [节点...]}，
                               也可直接给一层的节点列表（默认算在 'A' 层）
            rule             : 失效规则，签名 rule(net) -> 可迭代的 (层, 节点)；
                               默认 None = 布尔耦合（即 L0 的规则）
            max_steps        : 安全上限，防止规则写错导致死循环
            record           : True 时把每一轮的 S 记进 self.history

        返回:
            dict: {'S_A', 'S_B', 'S_mean', 'steps', 'failed_initial', 'history'}

        ⚠️ 每次调用前会 reset()，所以同一个对象可以反复用来做多次实验。
        """
        self.reset()
        if isinstance(initial_failures, dict):
            init = {layer: list(v) for layer, v in initial_failures.items()}
        else:
            init = {"A": list(initial_failures), "B": []}

        n_init = 0
        for layer, nodes in init.items():
            for n in nodes:
                if self.alive[layer][n]:
                    self.kill(layer, n)
                    n_init += 1

        if record:
            self.history = [{"step": 0, "S_A": self.S("A"), "S_B": self.S("B")}]

        for step in range(1, max_steps + 1):
            newly = set(rule(self)) if rule is not None else set(boolean_coupling_rule(self))
            # 规则可能重复报同一个节点，这里统一过滤掉已经失效的
            newly = {(l, n) for l, n in newly if self.alive[l][n]}

            pruned = {(layer, n) for layer in ("A", "B") for n in self.prune_gcc(layer)}

            if not newly and not pruned:
                self.steps = step - 1
                break

            for layer, node in newly:
                self.kill(layer, node)

            if record:
                self.history.append(
                    {"step": step, "S_A": self.S("A"), "S_B": self.S("B")}
                )
        else:
            self.steps = max_steps

        return {
            "S_A": self.S("A"),
            "S_B": self.S("B"),
            "S_mean": 0.5 * (self.S("A") + self.S("B")),
            "steps": self.steps,
            "failed_initial": n_init,
            "history": list(self.history),
        }


def boolean_coupling_rule(net):
    """
    L0 的失效规则：**布尔耦合** —— 我依赖的伙伴失效了，我也失效。

    这是最朴素的层间依赖：没有强度、没有容差，要么活着要么死（0/1 布尔）。
    只跨层传播；"与主干断开"由骨架里的 prune_gcc 负责。
    """
    newly = []
    for layer, other in (("A", "B"), ("B", "A")):
        for node, partner in net.partner[layer].items():
            if net.alive[layer][node] and not net.alive[other][partner]:
                newly.append((layer, node))
    return newly
