"""
韧性度量工具箱 —— p_c / 相变阶数 / 韧性损失面积 AUC / RHI / 恢复过程 / 脆弱性探针

【这个文件为什么存在】
  有了 phase6 的四级模型，我们能造出 S(f) 曲线（f = 初始移除比例，S = 存活序参量）。
  但"曲线"本身不是结论：**要把它压成数字才能比较、才能写进论文表格**。
  这个文件就是那套"压成数字"的标准动作，口径统一后实验才可复现、可对拍。

【知识地图（每条都是可查文献的既有口径，别自己发明名字）】
  1. 渗流阈值 p_c
     网络崩到什么程度就"垮了"的分界点。两种常用操作定义（本文件都实现）：
       · 中位数法：S 第一次跌破 0.5 的 f（平滑曲线用，二阶相变）
       · 最大跌幅法：S 跌得最猛的那一步（有跳变时更稳，一阶相变）
     本项目建议：**两种都算，报的时候写清用哪种**（论文里最常见的争议就出在这）。
  2. 相变阶数（一阶 / 二阶）
     按最大单步跌幅判断：跌幅大 ⇒ 一阶（突然崩塌）；连续下降 ⇒ 二阶。
  3. 韧性损失面积（resilience loss area, AUC 型）
     口径出处：**Ouyang, Dueñas-Osorio & Min, Structural Safety 25:33 (2012)**
     把韧性曲线与"满分线 S=1"之间的面积记为损失：
         LossArea = ∫ (1 − S(f)) df
     ⚠️ 归因提醒：常被误引成 Ouyang & Wang 2015 RESS——那篇讲的是相依系统联合恢复，
        韧性损失面积的出处是 **2012 Structural Safety**。
  4. RHI（resilience hybrid index，混合韧性指数）—— 档 B 的预测目标
     RHI = 曲线下面积 / 区间长度 = 1 − LossArea/区间长度 ∈ [0, 1]，越大越韧性。
     它和 LossArea 是同一个数的两种说法，报的时候别两个都当独立指标列。
  5. 恢复过程指标（restoration）
     恢复斜率（最快恢复速率）、恢复时间（回到基线/目标需要多久）。
     出处参见 Hosseini/Barker/Ramirez-Marquez, RESS 145:47 (2016)（70+ 韧性指标综述）。
  6. 脆弱性探针（vulnerability probe）
     单点级联后果 = 移除此节点前后 p_c 的变化：
         I_i = p_c⁰ − p_cⁱ
     ⚠️ 这是本项目对 FINDER 资产的**角色定位**：FINDER 会"拆"网络，
        本项目要"防"——所以这里只把单点后果当**脆弱性探针**，不作攻击算法。

【怎么用（最小例子）】
    from metrics.resilience_metrics import loss_area, rhi, percolation_threshold
    f, S = percolation_curve(G_A, G_B)          # phase6 的 L0 产出
    print(percolation_threshold(f, S), loss_area(f, S), rhi(f, S))

运行自检（不依赖 pytest，直接跑本文件即可）：
    python src/metrics/resilience_metrics.py
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "jump_size", "max_slope", "transition_order",
    "threshold_by_midpoint", "threshold_by_max_drop", "percolation_threshold",
    "loss_area", "rhi", "recovery_slope", "recovery_time", "recovery_metrics",
    "vulnerability_probe",
]


# ============================================================
# 相变：阶数与阈值
# ============================================================

def jump_size(S):
    """最大单步跌幅（相变判据用）。S 可以是 list / numpy 数组。"""
    S = np.asarray(S, dtype=float)
    if len(S) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(S))))


def max_slope(fractions, S):
    """
    最大斜率 |dS/df|（每单位 f 的最大跌幅）。

    为什么要有它：jump_size 是"相邻两点的差"，**会随 f 网格密度变化**——
    同一组实验网格密一倍，jump 就大概小一半，于是同一个物理过程可能一会儿被
    判成一阶、一会儿被判成二阶。跨研究比较相变剧烈程度时，请用本函数
    （单位一致），而不要比 jump_size。
    """
    f = np.asarray(fractions, dtype=float)
    S = np.asarray(S, dtype=float)
    if len(f) < 2:
        return 0.0
    df = np.diff(f)
    if np.any(df <= 0):
        raise ValueError("fractions 必须严格递增")
    return float(np.max(np.abs(np.diff(S) / df)))


def transition_order(fractions, S, jump_threshold=0.2):
    """
    一阶 or 二阶（操作定义，口径与 phase6/L0_static_dependency.py 保持一致）。

        max_jump > jump_threshold  →  '一阶'（有跳变）
        否则                        →  '二阶'（连续下降）

    ⚠️ jump_tag 是"相邻两点的差"，**依赖 f 网格密度**：网格加密后同一个跳变
       会被切成好几步，jump 变小、判据可能翻转。所以：
         · 判阶数时保持同一套 f 网格（本项目统一 linspace(0, 0.5, 26)）；
         · 要跨研究比较"崩得多猛"，用 max_slope（单位跌幅）而不是 jump_size。
    """
    jump = jump_size(S)
    if len(np.asarray(S)) < 2:
        return "数据不足", jump
    return ("一阶" if jump > jump_threshold else "二阶"), jump


def _interp_crossing(fractions, S, level):
    """在相邻两点之间线性插值，求 S 首次跌破 level 的 f。找不到返回 None。"""
    f = np.asarray(fractions, dtype=float)
    S = np.asarray(S, dtype=float)
    for i in range(len(S) - 1):
        if S[i] >= level > S[i + 1]:
            # 线性插值：这段里 S 从 S[i] 掉到 S[i+1]
            ratio = (S[i] - level) / (S[i] - S[i + 1]) if S[i] != S[i + 1] else 0.0
            return float(f[i] + ratio * (f[i + 1] - f[i]))
    if S[-1] <= level:
        return float(f[-1])
    return None


def threshold_by_midpoint(fractions, S, level=0.5):
    """中位数法：S 跌破 level（默认 0.5）的 f。适合平滑（二阶）曲线。"""
    return _interp_crossing(fractions, S, level)


def threshold_by_max_drop(fractions, S):
    """最大跌幅法：取 S 跌得最猛一步的 f 中点。有跳变（一阶）时最稳。"""
    f = np.asarray(fractions, dtype=float)
    S = np.asarray(S, dtype=float)
    if len(S) < 2:
        return None
    k = int(np.argmax(np.abs(np.diff(S))))
    return float(0.5 * (f[k] + f[k + 1]))


def percolation_threshold(fractions, S, method="auto", level=0.5,
                          jump_threshold=0.2):
    """
    统一入口：给出 p_c。

    method = 'midpoint' | 'max_drop' | 'auto'
    auto：先判相变阶数——一阶用最大跌幅法，二阶用中位数法。
    """
    if method == "midpoint":
        return threshold_by_midpoint(fractions, S, level=level)
    if method == "max_drop":
        return threshold_by_max_drop(fractions, S)
    if method == "auto":
        order, _ = transition_order(fractions, S, jump_threshold=jump_threshold)
        if order == "一阶":
            return threshold_by_max_drop(fractions, S)
        return threshold_by_midpoint(fractions, S, level=level)
    raise ValueError(f"不认识的 method: {method}")


# ============================================================
# 韧性面积 / RHI
# ============================================================

def loss_area(fractions, S):
    """
    韧性损失面积（AUC 型，Ouyang 2012 口径）：

        LossArea = ∫ (1 − S(f)) df     ← 曲线与"满分线 S=1"之间的面积

    值越大 = 掉得越多/越久 = 越不韧性。单位是"f 的单位 × 1"。
    """
    f = np.asarray(fractions, dtype=float)
    S = np.asarray(S, dtype=float)
    if len(f) != len(S):
        raise ValueError("fractions 与 S 长度必须一致")
    if len(f) < 2:
        return 0.0
    return float(np.trapezoid(1.0 - S, f))


def rhi(fractions, S):
    """
    RHI（混合韧性指数）= 曲线下面积 / 区间长度 = 1 − LossArea/区间长度 ∈ [0, 1]。
    越大越韧性。这是档 B（预测 RHI）要预测的那个量。
    """
    f = np.asarray(fractions, dtype=float)
    S = np.asarray(S, dtype=float)
    span = float(f[-1] - f[0])
    if span <= 0:
        return float(np.mean(S))
    return float(1.0 - loss_area(f, S) / span)


# ============================================================
# 恢复过程
# ============================================================

def recovery_slope(times, performance):
    """恢复斜率 = 性能上升阶段的最大斜率（单位时间恢复速率）。"""
    t = np.asarray(times, dtype=float)
    P = np.asarray(performance, dtype=float)
    if len(t) < 2:
        return 0.0
    slopes = np.diff(P) / np.diff(t)
    return float(np.max(slopes)) if len(slopes) else 0.0


def recovery_time(times, performance, target=1.0):
    """
    恢复时间：性能**从谷底爬回** target 的最早时刻；一直没回到 ⇒ None。

    实现细节（踩过坑）：必须从**最低点之后**开始找。
    否则 t=0 时性能本来就是 1.0，会直接返回 0，等于永远"已经恢复了"。

    注意（容易混）：韧性损失面积是"降级期间累计损失"，恢复时间是"要多久爬回来"，
    两者是不同维度，别用一个代替另一个。
    """
    t = np.asarray(times, dtype=float)
    P = np.asarray(performance, dtype=float)
    if len(P) == 0:
        return None
    trough = int(np.argmin(P))          # 谷底下标：从它之后才开始数恢复
    hit = np.where(P[trough:] >= target)[0]
    if len(hit) == 0:
        return None
    return float(t[trough + hit[0]])


def recovery_metrics(times, performance, target=1.0):
    """一次给出恢复过程的三个常用数：起点性能、恢复斜率、恢复时间。"""
    P = np.asarray(performance, dtype=float)
    return {
        "min_performance": float(np.min(P)),
        "recovery_slope": recovery_slope(times, performance),
        "recovery_time": recovery_time(times, performance, target=target),
    }


# ============================================================
# 脆弱性探针
# ============================================================

def vulnerability_probe(pc_baseline, pc_without_node):
    """
    单点级联后果（脆弱性探针）：

        I_i = p_c⁰ − p_cⁱ

    p_c⁰ = 原始网络的渗流阈值，p_cⁱ = 去掉节点 i 后的渗流阈值。
    I_i 越大 ⇒ 移除该节点让网络更早崩塌 ⇒ 它越"关键"。

    ⚠️ 角色口径：FINDER 那类方法是"拆"（找要移除的节点），
       本项目是"防"（找要加固的节点）。所以这里只把它当作**脆弱性探针**，
       用来给"关键节点集合"定位，而不是当作攻击策略发表。
    """
    if pc_baseline is None or pc_without_node is None:
        return None
    return float(pc_baseline - pc_without_node)


# ============================================================
# 自检（直接运行本文件时执行，不需要 pytest）
# ============================================================

def _self_check():
    f = np.linspace(0.0, 1.0, 11)

    # 1) 满分曲线：无损失、RHI = 1
    perfect = np.ones_like(f)
    assert abs(loss_area(f, perfect)) < 1e-12, "满分线不该有损失面积"
    assert abs(rhi(f, perfect) - 1.0) < 1e-12, "满分线 RHI 应为 1"

    # 2) 全灭曲线：损失面积 = 区间长度，RHI = 0
    dead = np.zeros_like(f)
    assert abs(loss_area(f, dead) - 1.0) < 1e-12, "全灭曲线损失面积应等于区间长度"
    assert abs(rhi(f, dead)) < 1e-12, "全灭曲线 RHI 应为 0"

    # 3) 中位数法阈值：S 在 f=0.5 处正好穿过 0.5
    S_mid = 1.0 - f
    assert abs(threshold_by_midpoint(f, S_mid, 0.5) - 0.5) < 1e-9

    # 4) 最大跌幅法：S 在 f=0.4→0.5 之间从 1 掉到 0 ⇒ 阈值取中点 0.45
    S_jump = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    assert abs(threshold_by_max_drop(f, S_jump) - 0.45) < 1e-9
    order, jump = transition_order(f, S_jump)
    assert order == "一阶" and jump > 0.9

    # 5) RHI 与 LossArea 互为补数（同一信息、两种说法）
    S_linear = np.linspace(1.0, 0.0, 11)
    assert abs(rhi(f, S_linear) - (1.0 - loss_area(f, S_linear))) < 1e-12

    # 6) 恢复指标：性能 1 → 0.2 →（t=3 起）回到 1
    times = [0, 1, 2, 3, 4]
    perf = [1.0, 0.2, 0.2, 1.0, 1.0]
    m = recovery_metrics(times, perf)
    assert abs(m["min_performance"] - 0.2) < 1e-12
    assert abs(m["recovery_slope"] - 0.8) < 1e-12
    assert abs(m["recovery_time"] - 3.0) < 1e-12

    # 7) 脆弱性探针：阈值变小（提前崩塌）⇒ 正数
    assert abs(vulnerability_probe(0.4, 0.25) - 0.15) < 1e-12

    print("metrics.resilience_metrics 自检通过（7 组断言）")


if __name__ == "__main__":
    _self_check()
