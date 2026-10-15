# notebooks · 边跑边学（7 个）

从零学这套内核的顺序：**先跑代码看现象，再回读源码**。每个 notebook 都是
"环境准备 → 分步演示 → 思考题"的结构，代码单元全部经过等价执行验证（见下）。

| # | Notebook | 对应代码 | 学到什么 |
|---|---|---|---|
| 01 | `01_骨架_双层相依网络.ipynb` | `phase6_interdependent/core.py` | 双层网络怎么表示、级联为什么是"求不动点"、层间依赖为什么致命 |
| 02 | `02_L0_静态结构依赖与相变.ipynb` | `L0_static_dependency.py` | 布尔耦合 + GCC；为什么层间耦合把二阶相变变一阶 |
| 03 | `03_L1_网内动力学与容忍带.ipynb` | `L1_intra_layer_dynamics.py` | 自动力学 `M(x)`、容忍判据 δ、**δ=1 退回 L0** 的退化性检验 |
| 04 | `04_L2_状态驱动耦合Wt.ipynb` | `L2_state_driven_coupling.py` | `W(t)` 状态驱动耦合、满足度 φ、**ρ=0∧φ=0 退回 L0** |
| 05 | `05_L3_超图逻辑依赖.ipynb` | `L3_hypergraph_logic.py` | 超边与 AND/OR/k-of-N 语义、功能度量 F、**二元 AND 退回 L0** |
| 06 | `06_韧性度量与实验设计.ipynb` | `src/metrics/` | 曲线 → p_c / 损失面积 / RHI / 恢复指标 / 脆弱性探针 |
| 07 | `07_CPS主循环_耦合张量.ipynb` | `cps_cascade.py` | 两张耦合张量 `W^C`/`W^S`、δ/φ/Φ 三判据、新失效通道（只退化不删点）、**关掉判据退回 L0** 的退化性检验 |

## 怎么运行

本机**当前没装 jupyter**（`jupyter / nbformat / nbconvert / ipykernel` 都未安装），所以有两种方式：

```bash
# 方式一（推荐）：装一个最小内核，然后在 VS Code 或 Jupyter 里打开
pip install jupyter ipykernel

# 方式二：不装也行 —— notebook 的代码单元就是普通 Python，
#         复制到临时脚本里跑即可（每个 notebook 的前两格必须一起跑）
```

## 这些 notebook 是怎么验证过的

写 notebook 时本机没有 jupyter，所以用**等价执行**验证：把每个 `.ipynb` 的所有 code cell 拼成
一个脚本，在项目根目录下 `exec`（`matplotlib` 用 Agg 后端）。7 个 notebook 全部通过。

结论：**代码是对的，只是渲染层没在本机跑过**。首次在 Jupyter 里打开时如果遇到路径问题，
先看第一格（它会从当前目录往上找到 `src/phase6_interdependent`）。

## 注意

- 每个 notebook 的前两格（环境准备 + 公共底座）必须**一起执行**，否则 `G_A / G_B` 不存在。
- 图会存到 `results/`（文件名形如 `notebook_0X_*.png`）。
- 所有随机过程都固定了种子，重跑结果一致；想换随机实例就改 `seed`。
