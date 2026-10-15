# metrics · 韧性度量工具箱

曲线（S–f）本身不是结论，**压成数字才能比较、才能进论文表格**。本包就是那套"压成数字"的标准动作。
放在 `src/` 顶层而不是某个 phase 里，是因为 L0–L3 四级台阶、phase4 稳健性、phase5 级联都要用同一套口径——
**口径只能有一份**。

## 指标清单

| 函数 | 含义 | 口径出处 | 注意 |
|---|---|---|---|
| `jump_size(S)` | 相邻两点最大差 | —— | **依赖 f 网格密度**，只用于同一网格内的判阶 |
| `max_slope(f, S)` | 每单位 f 的最大跌幅 \|dS/df\| | —— | 跨研究比较"崩得多猛"用这个 |
| `transition_order(f, S)` | 一阶 / 二阶 | 本项目操作定义 | 与 `L0_static_dependency.transition_order` 同口径 |
| `threshold_by_midpoint(f, S, level=0.5)` | S 首次跌破 level 的 f | 常见做法 | 适合平滑（二阶）曲线 |
| `threshold_by_max_drop(f, S)` | 跌得最猛一步的 f 中点 | 常见做法 | 有跳变（一阶）时更稳 |
| `percolation_threshold(f, S, method="auto")` | 统一入口（自动选口径） | —— | 报 p_c 必须写清用了哪种 |
| `loss_area(f, S)` | 韧性损失面积 = ∫(1−S)df | **Ouyang et al., Structural Safety 25:33 (2012)** | AUC 型，值越大越不韧性 |
| `rhi(f, S)` | 混合韧性指数 = 1 − 损失面积/区间 ∈ [0,1] | 档 B 的预测目标 | 与损失面积**同源**，别当两个创新点 |
| `recovery_slope(t, P)` | 恢复阶段最大上升斜率 | Hosseini et al., RESS 145:47 (2016) | —— |
| `recovery_time(t, P, target)` | 从**谷底**爬回 target 的时刻 | 同上 | 从谷底起算，否则恒为 0 |
| `recovery_metrics(t, P)` | 上面三个数的打包 | —— | —— |
| `vulnerability_probe(pc0, pc1)` | 单点级联后果 I_i = p_c⁰ − p_cⁱ | 本项目口径 | FINDER 资产的**脆弱性探针**定位，非攻击算法 |

## 怎么用

```python
from metrics.resilience_metrics import percolation_threshold, loss_area, rhi
f, S = percolation_curve(G_A, G_B)          # phase6/L0 的产出
print(percolation_threshold(f, S, method="auto"), loss_area(f, S), rhi(f, S))
```

## 四个必须记住的口径纪律

1. **p_c 有两种口径，报的时候写清**：一阶相变时"中位数法"和"最大跌幅法"差得最多。
2. **损失面积与 RHI 是同一个数的两种说法**：`RHI = 1 − LossArea/区间长度`，不能当两个独立指标。
3. **判阶不要用不同网格的 jump 比较**：网格加密 jump 会变小，判据可能翻转；跨研究比陡峭度用 `max_slope`。
4. **归因别混**：韧性损失面积出自 **Ouyang 2012 Structural Safety**，不是 Ouyang & Wang 2015 RESS。

## 自检

```bash
python src/metrics/resilience_metrics.py     # 7 组断言，不需要 pytest
python -m pytest tests/test_metrics.py -q    # 25 项测试
```
