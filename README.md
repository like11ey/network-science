# 复杂网络级联失效仿真平台

**Complex Network Cascading Failure Simulation Platform**

从Python零基础到完成导师课题的完整学习项目。涵盖Python基础、网络分析、鲁棒性评估、级联失效仿真、相依网络分析、数据可视化和机器学习集成。

> **⭐ 课题依托（2026-08-30）：** 本平台是导师国自然面上项目《信息-物理耦合网络视角下高端装备制造系统韧性涌现机理与主动可靠性管理研究》（2027.01–2030.12）的**仿真内核**：**phase6 相依网络级联 = 信息-物理耦合网络内核（最高优先级）**，**phase5 负载-容量级联 = 设备退化级联机制**。详细对齐见根目录 `规划\科研主线（信息-物理耦合网络）.md`（总纲）与本目录 CLAUDE.md 顶部"项目语境"。

---

## 项目特色

- **8个阶段、渐进式学习**：从Python语法到前沿研究方法
- **理论+代码+练习**：每个文件都有详细中文注释和动手练习
- **对接真实研究**：对应复杂网络鲁棒性与级联失效方向的核心论文
- **面试友好**：涵盖Python面试高频考点和项目经验展示

---

## 目录结构

```
ai_python/
├── README.md                          # 项目说明（本文件）
├── CLAUDE.md                          # 详细学习指南
├── requirements.txt                   # Python依赖
├── setup.py                           # 包安装配置
│
├── src/                               # 源代码
│   ├── __init__.py
│   ├── phase1_fundamentals/           # Phase 1: Python基础
│   │   ├── 01_variables_and_types.py  # 变量与数据类型
│   │   ├── 02_control_flow.py         # 条件语句与循环
│   │   ├── 03_data_structures.py      # 列表、字典、集合、元组
│   │   ├── 04_functions.py            # 函数、lambda、闭包、装饰器
│   │   ├── 05_file_io.py              # 文件读写（CSV/JSON/Pickle）
│   │   ├── 06_oop.py                  # 面向对象编程
│   │   └── 07_advanced_features.py    # 高级特性（推导式、生成器、异步）
│   │
│   ├── phase2_network_basics/         # Phase 2: NetworkX基础
│   │   └── 01_networkx_basics.py      # 网络创建、度量、中心性分析
│   │
│   ├── phase3_network_models/         # Phase 3: 经典网络模型
│   │   └── 01_four_models.py          # 规则/ER/WS/BA四种模型对比
│   │
│   ├── phase4_robustness/             # Phase 4: 网络鲁棒性
│   │   └── 01_robustness_analysis.py  # 随机故障与蓄意攻击
│   │
│   ├── phase5_cascading/              # Phase 5: 级联失效
│   │   └── 01_cascading_failure.py    # 负载-容量模型仿真
│   │
│   ├── phase6_interdependent/         # Phase 6: 相依网络
│   │   └── 01_interdependent_cascading.py  # 双层耦合网络级联
│   │
│   ├── phase7_visualization/          # Phase 7: 数据可视化
│   │   └── 01_network_viz.py          # 网络图、仪表板、分析报告
│   │
│   └── phase8_ml_integration/         # Phase 8: 机器学习集成
│       └── 01_ml_prediction.py        # 节点重要性预测（RF/GB/LR）
│
├── data/                              # 数据文件
│   ├── edges.csv                      # 边列表（CSV格式）
│   ├── network.json                   # 网络数据（JSON格式）
│   ├── ba_network.gml                 # BA网络（GML格式）
│   └── ...
│
├── results/                           # 输出图表
│   ├── degree_distribution.png        # 度分布对比
│   ├── network_visualization.png      # 网络结构可视化
│   ├── er_vs_ba_robustness.png        # 鲁棒性对比
│   ├── cascading_alpha.png            # 级联失效（α参数）
│   ├── interdependent_percolation.png # 相依网络渗流
│   ├── ml_model_comparison.png        # ML模型对比
│   └── ...
│
└── tests/                             # 测试文件
    └── test_network.py
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目
cd ai_python

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import networkx, numpy, pandas, matplotlib, sklearn; print('环境OK')"
```

### 2. 逐步运行

```bash
# Phase 1: Python基础
python src/phase1_fundamentals/01_variables_and_types.py
python src/phase1_fundamentals/02_control_flow.py
python src/phase1_fundamentals/03_data_structures.py
python src/phase1_fundamentals/04_functions.py
python src/phase1_fundamentals/05_file_io.py
python src/phase1_fundamentals/06_oop.py
python src/phase1_fundamentals/07_advanced_features.py

# Phase 2: NetworkX基础
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

---

## 各阶段概览

| 阶段 | 主题 | 核心内容 | 对应研究 | 预计时间 |
|------|------|----------|----------|----------|
| Phase 1 | Python基础 | 变量、控制流、数据结构、函数、OOP、文件IO、高级特性 | — | 5-7天 |
| Phase 2 | NetworkX | 网络创建、度量计算、中心性分析、邻接矩阵 | PPT第2-3章 | 3-4天 |
| Phase 3 | 网络模型 | 规则/ER/WS/BA模型、度分布、幂律检验 | PPT第4章 | 3-4天 |
| Phase 4 | 鲁棒性 | 随机故障、蓄意攻击、渗流理论、Malloy-Reed判据 | Albert et al. (2000) | 4-5天 |
| Phase 5 | 级联失效 | 负载-容量模型、负载分配策略、可调参数β | Duan et al. | 5-7天 |
| Phase 6 | 相依网络 | 双层耦合、级联崩溃、一级/二级相变 | Duan et al., PNAS 2019 | 4-5天 |
| Phase 7 | 可视化 | 网络图、度分布图、仪表板、级联过程动画 | — | 2-3天 |
| Phase 8 | ML集成 | 特征提取、随机森林、梯度提升、模型评估 | — | 3-4天 |

---

## 学习成果检验

完成本项目后，你将掌握：

### Python编程能力
- [ ] 熟练使用Python数据结构（list, dict, set, tuple）
- [ ] 掌握函数式编程（lambda, 装饰器, 生成器）
- [ ] 面向对象编程（继承、多态、dataclass）
- [ ] 文件读写（CSV, JSON, Pickle）
- [ ] 列表推导式、上下文管理器等高级特性

### 网络科学知识
- [ ] 理解度分布、集聚系数、中心性等核心概念
- [ ] 掌握ER、WS、BA等经典网络模型
- [ ] 理解无标度网络的"鲁棒且脆弱"特性
- [ ] 掌握渗流理论和相变概念

### 工程实践能力
- [ ] 使用NetworkX进行网络分析
- [ ] 实现级联失效仿真（负载-容量模型）
- [ ] 构建相依网络模型
- [ ] 使用matplotlib生成专业图表
- [ ] 使用scikit-learn进行机器学习建模

### 研究方法论
- [ ] 能够复现经典论文的实验结果
- [ ] 能够设计对比实验并分析结果
- [ ] 能够将机器学习方法应用于网络分析

---

## 面试/简历使用指南

### 简历项目描述示例

> **复杂网络级联失效仿真平台** | Python, NetworkX, scikit-learn
>
> - 实现了ER、WS、BA等经典网络模型的生成与对比分析
> - 开发了负载-容量级联失效仿真器，支持多种负载分配策略
> - 构建了双层相依网络模型，验证了一级相变现象
> - 使用随机森林/梯度提升模型预测网络关键节点，F1分数达到0.95+
> - 生成了20+张专业分析图表，包括度分布、鲁棒性曲线、混淆矩阵等

### 面试常见问题准备

1. **什么是无标度网络？它和随机网络有什么区别？**
   → 运行Phase 3，观察度分布图

2. **为什么无标度网络对随机故障鲁棒但对蓄意攻击脆弱？**
   → 运行Phase 4，理解Malloy-Reed判据

3. **什么是级联失效？如何建模？**
   → 运行Phase 5，理解负载-容量模型

4. **相依网络和单层网络的鲁棒性有什么区别？**
   → 运行Phase 6，理解一级相变vs二级相变

5. **如何识别网络中的关键节点？**
   → 运行Phase 8，理解特征工程和机器学习方法

---

## 依赖说明

```
networkx>=3.0      # 网络分析核心库
numpy>=1.24        # 数值计算
pandas>=2.0        # 数据处理
matplotlib>=3.7    # 数据可视化
seaborn>=0.12      # 统计图表
scikit-learn>=1.3  # 机器学习
scipy>=1.10        # 科学计算
jupyter>=1.0       # 交互式开发（可选）
```

---

## 参考文献

1. Albert, R., Jeong, H., & Barabási, A. L. (2000). Error and attack tolerance of complex networks. *Nature*, 406(6794), 378-382.
2. Duan, D., et al. (2019). Universal behavior of cascading failures in interdependent networks. *PNAS*, 116(45), 22472-22477.
3. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
4. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
