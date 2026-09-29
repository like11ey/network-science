# 复杂网络级联失效仿真平台

> **本文件只写本项目内容。** 用户画像 / 研究方向口径 / 协作硬规则 / 机器环境见全局正本 `C:\Users\Admin\AI-Memory\GLOBAL_MEMORY.md`；上层项目记忆：`E:\KeYan\Academic Roadmap\AGENTS.md`。

Complex Network Cascading Failure Simulation Platform — 从 Python 零基础到完成导师课题的完整学习项目。

## 项目概述

本项目实现了一个完整的复杂网络级联失效仿真平台，涵盖 8 个渐进式学习阶段：Python 基础 → 网络分析 → 经典模型 → 鲁棒性评估 → 级联失效仿真 → 相依网络 → 可视化 → 机器学习集成。核心研究方向对应 Albert et al. (2000) Nature 和 Duan et al. (2019) PNAS 等经典论文。

> **⭐ 课题依托（2026-08-30）：** 本平台服务于导师国自然面上项目《信息-物理耦合网络视角下高端装备制造系统韧性涌现机理与主动可靠性管理研究》（2027.01–2030.12）：**phase6 相依网络级联 = 信息-物理耦合网络仿真内核（最高优先级，研一下起升级为动态耦合强度+超图数据引擎）**；**phase5 负载-容量级联 = 设备退化→过载→崩溃机制**。总纲：根目录 `规划\科研主线（信息-物理耦合网络）.md`；分 Phase 优先级见 CLAUDE.md 顶部。

## 技术栈

- **Python 3.11** — 主要开发语言
- **NetworkX** — 网络分析与图算法
- **NumPy / SciPy** — 数值计算与科学计算
- **Matplotlib / Seaborn** — 数据可视化
- **scikit-learn** — 机器学习建模
- **pytest** — 单元测试

## 目录结构

```
项目根目录/
├── src/                  # 源代码（8 个阶段子包）
├── data/                 # 数据文件（运行时自动生成）
├── results/              # 输出图表（运行时自动创建）
├── tests/                # 测试文件
├── requirements.txt      # Python 依赖
├── setup.py              # 包安装配置
├── CLAUDE.md             # 详细学习指南
└── README.md             # 项目说明
```

## 8 个阶段模块

### Phase 1: Python 基础 (`src/phase1_fundamentals/`)

| 文件 | 内容 |
| ------ | ------ |
| `01_variables_and_types.py` | 变量、数据类型、f-string、类型转换 |
| `02_control_flow.py` | 条件语句、循环、列表推导式 |
| `03_data_structures.py` | list/dict/set/tuple、Counter、defaultdict |
| `04_functions.py` | 函数、lambda、闭包、装饰器、生成器 |
| `05_file_io.py` | CSV/JSON/Pickle 读写、边列表格式 |
| `06_oop.py` | 面向对象、继承、@property、@dataclass |
| `07_advanced_features.py` | 推导式、迭代器、上下文管理器、itertools |

### Phase 2: NetworkX 基础 (`src/phase2_network_basics/`)

| 文件 | 内容 |
|------|------|
| `01_networkx_basics.py` | 网络创建、度/集聚系数/最短路径/介数、邻接矩阵、导出网络 |

### Phase 3: 四种经典网络模型 (`src/phase3_network_models/`)

| 文件 | 内容 |
|------|------|
| `01_four_models.py` | 规则网络、ER 随机网络、WS 小世界、BA 无标度网络对比 |

### Phase 4: 网络鲁棒性分析 (`src/phase4_robustness/`)

| 文件 | 内容 |
|------|------|
| `01_robustness_analysis.py` | 随机故障/蓄意攻击、渗流模拟、Malloy-Reed 判据 |

### Phase 5: 级联失效仿真 (`src/phase5_cascading/`)

| 文件 | 内容 |
|------|------|
| `01_cascading_failure.py` | 负载-容量模型、`CascadingFailureSimulator` 类、可调参数 β |

### Phase 6: 相依网络级联失效 (`src/phase6_interdependent/`)

| 文件 | 内容 |
|------|------|
| `01_interdependent_cascading.py` | 双层耦合网络、`InterdependentNetwork` 类、一级/二级相变 |

### Phase 7: 网络可视化 (`src/phase7_visualization/`)

| 文件 | 内容 |
|------|------|
| `01_network_viz.py` | 基础网络图、综合分析仪表板、度分布图 |

### Phase 8: 机器学习集成 (`src/phase8_ml_integration/`)

| 文件 | 内容 |
|------|------|
| `01_ml_prediction.py` | 节点特征提取、随机森林/梯度提升/逻辑回归分类、模型评估 |

## 运行命令

### 运行各阶段模块

```bash
# Phase 1: Python 基础
python src/phase1_fundamentals/01_variables_and_types.py
python src/phase1_fundamentals/02_control_flow.py
python src/phase1_fundamentals/03_data_structures.py
python src/phase1_fundamentals/04_functions.py
python src/phase1_fundamentals/05_file_io.py
python src/phase1_fundamentals/06_oop.py
python src/phase1_fundamentals/07_advanced_features.py

# Phase 2: NetworkX 基础
python src/phase2_network_basics/01_networkx_basics.py

# Phase 3: 网络模型
python src/phase3_network_models/01_four_models.py

# Phase 4: 鲁棒性分析
python src/phase4_robustness/01_robustness_analysis.py

# Phase 5: 级联失效
python src/phase5_cascading/01_cascading_failure.py

# Phase 6: 相依网络
python src/phase6_interdependent/01_interdependent_cascading.py

# Phase 7: 可视化
python src/phase7_visualization/01_network_viz.py

# Phase 8: 机器学习
python src/phase8_ml_integration/01_ml_prediction.py
```

### 运行测试

```bash
# 运行全部测试（详细输出）
python -m pytest tests/test_network.py -v

# 运行全部测试（简略输出）
python -m pytest tests/test_network.py

# 只运行某个测试函数
python -m pytest tests/test_network.py -v -k "test_er_network_avg_degree"

# 第一个失败就停止
python -m pytest tests/test_network.py -x

# 显示 print 输出
python -m pytest tests/test_network.py -v -s
```

> **注意**：必须使用 `python -m pytest` 而非直接调用 `pytest`，以确保使用当前虚拟环境的解释器。

### 环境验证

```bash
python -c "import networkx, numpy, pandas, matplotlib, sklearn; print('环境OK')"
```

## 路径约定

项目使用 `pathlib.Path` 动态解析路径，所有文件路径基于 `Path(__file__)` 自动定位项目根目录，不依赖硬编码绝对路径。

- **项目根目录**：`Path(__file__).resolve().parent.parent.parent`（从 `src/phaseX_xxx/` 回溯三级）
- **数据目录**：`PROJECT_ROOT / "data"`
- **输出目录**：`PROJECT_ROOT / "results"`

每个需要写入目录的模块文件顶部都有 `RESULTS_DIR.mkdir(exist_ok=True)` 或 `DATA_DIR.mkdir(exist_ok=True)` 自动创建目录。

## `results/` 输出目录

所有图表输出保存到 `results/` 目录（运行时自动创建）：

| 文件 | 来源模块 | 说明 |
| ------ | ---------- | ------ |
| `degree_distribution.png` | Phase 3 | 四种模型度分布对比 |
| `network_visualization.png` | Phase 3 | 四种网络结构可视化 |
| `er_robustness.png` | Phase 4 | ER 网络鲁棒性曲线 |
| `ba_robustness.png` | Phase 4 | BA 网络鲁棒性曲线 |
| `er_vs_ba_robustness.png` | Phase 4 | ER vs BA 鲁棒性对比 |
| `cascading_alpha.png` | Phase 5 | 容量参数 α 对级联的影响 |
| `cascading_beta.png` | Phase 5 | 可调参数 β 的影响 |
| `interdependent_percolation.png` | Phase 6 | 相依网络渗流曲线 |
| `coupling_strength.png` | Phase 6 | 耦合强度影响 |
| `ba_network.png` | Phase 7 | BA 网络基础图 |
| `er_network.png` | Phase 7 | ER 网络基础图 |
| `ba_dashboard.png` | Phase 7 | BA 综合分析仪表板 |
| `er_dashboard.png` | Phase 7 | ER 综合分析仪表板 |
| `ml_model_comparison.png` | Phase 8 | ML 模型性能对比 |
| `ml_confusion_matrices.png` | Phase 8 | ML 混淆矩阵 |

## `data/` 数据目录

数据文件由脚本运行时自动生成：

| 文件 | 来源模块 | 说明 |
| ------ | ---------- | ------ |
| `sample_network.txt` | Phase 1 (05) | 简单网络文本数据 |
| `edges.csv` | Phase 1 (05) | 带权边列表（CSV 格式） |
| `network.json` | Phase 1 (05) | 完整网络描述（JSON 格式） |
| `analysis_result.pkl` | Phase 1 (05) | Python 对象序列化 |
| `ba_network.gml` | Phase 2 (01) | BA 网络（GML 格式） |
| `ba_edges.txt` | Phase 2 (01) | BA 网络边列表 |
| `ba_adjlist.txt` | Phase 2 (01) | BA 网络邻接表 |
| `network_analysis.json` | Phase 2 (01) | 网络分析结果 |
| `temp/` | Phase 1 (07) | 临时目录 |

## 中文字体配置

所有使用 matplotlib 的模块统一配置：

```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

`Microsoft YaHei` 优先（Windows 必装，完整支持中文和 Unicode 减号 U+2212）。

## 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖：networkx>=3.0, numpy>=1.24, pandas>=2.0, matplotlib>=3.7, seaborn>=0.12, scikit-learn>=1.3, scipy>=1.10

测试依赖：pytest（需单独安装：`python -m pip install pytest`）
