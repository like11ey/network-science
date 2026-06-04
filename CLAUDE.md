# 复杂网络级联失效仿真平台 —— 从零到项目实战

> **你的背景**：有C/Java基础，Python零基础
> **你的目标**：学会Python → 完成导师课题 → 找到实习
> **学习方法**：不是跟着教程学，而是在项目中边做边学

---

## 环境确认

依赖已经安装好了，验证一下：

```bash
python -c "import networkx, numpy, pandas, matplotlib, scipy; print('环境OK')"
```

---

# Phase 1：Python基础（预计5-7天）

> **目标**：用C/Java的思维过渡到Python，重点掌握Python特有的语法
> **核心理念**：Python不是"简化版Java"，而是一套完全不同的编程哲学
> **每个文件结尾都有练习函数(`exercise_xxx()`)，一定要动手做！**

---

## Day 1：变量和类型 + 控制流

### 第一步：运行文件看输出
```bash
python src/phase1_fundamentals/01_variables_and_types.py
```

**观察这些Python与C/Java的不同点：**
- `n = 10` 不需要 `int n = 10;`（动态类型）
- `10 ** 100` 是10的100次方（C的int会溢出）
- `f"网络规模: {nodes}个节点"` 这就是f-string，类似Java的 `"foo" + var`
- `x, y = y, x` 交换两个变量不需要临时变量！
- `True / False` 首字母大写（Java是 `true/false` 小写）
- `None` 就是Java的 `null`

**动手练习（修改代码看效果）：**
1. 把 `N = 1000` 改成 `N = 5000`，平均度变了吗？
2. 把 `p = 0.01` 改成 `p = 0.05`，期望边数变多了还变少了？为什么？
3. 试试 `f"π取两位: {3.14159:.2f}"` ，`: .2f` 是什么意思？

### 第二步：运行控制流文件
```bash
python src/phase1_fundamentals/02_control_flow.py
```

**和C/Java的对比（重点看）：**
- `elif` 不是 `else if`
- `for i in range(5)` 相当于 `for(int i=0; i<5; i++)`
- `2 <= k <= 10` 这种链式比较C做不到
- **列表推导式**：`[x**2 for x in range(10)]` 一行代替for循环+append
- Python没有 `switch-case`，用字典映射替代

**动手练习：**
1. 把列表推导式改成传统for循环，体会一下少了多少行代码
2. 尝试 `[x for x in range(50) if x % 3 == 0]` —— 这会生成什么？
3. 修改 `exercise_ba_degree_sequence()` 中的 `m=3` 改成 `m=5`，最大度变了多少？

---

## Day 2：数据结构（重中之重！）

```bash
python src/phase1_fundamentals/03_data_structures.py
```

**这部分是Python的精髓，一定吃透：**

| Python | Java | 特点 |
|--------|------|------|
| `list` | `ArrayList` | 切片操作是杀手锏 |
| `dict` | `HashMap` | 最常用的数据结构 |
| `set` | `HashSet` | 集合运算特别适合网络分析 |
| `tuple` | 不可变List | 函数返回多个值 |
| `defaultdict` | — | Python独有，避免KeyError |

**必须掌握的5个操作：**
```python
# 1. 切片：取序列的一部分
nodes[:3]    # 前3个
nodes[-2:]   # 后2个
nodes[::-1]  # 反转

# 2. 字典推导式：统计度分布
degree_dist = {node: len(neighbors) for node, neighbors in adj.items()}

# 3. 集合运算：找共同邻居
common = neighbors_1 & neighbors_2  # 交集

# 4. Counter：统计频次
from collections import Counter
degree_dist = Counter(degrees)

# 5. defaultdict：避免KeyError
adj = defaultdict(list)
adj[u].append(v)  # 即使u不存在也不会报错
```

**动手练习：**
1. 自己写一个边列表，用 `defaultdict` 构建邻接表
2. 用 `Counter` 统计一段文本中每个字母出现的次数
3. 把 `exercise_adjacency_list()` 运行三遍，观察输出

---

## Day 3：函数的高级用法

```bash
python src/phase1_fundamentals/04_functions.py
```

**Python函数的独特之处：**
- **函数是一等公民**：可以赋值给变量、当参数传递、当返回值
- **lambda**：匿名函数，`lambda x: x**2` 等价于 `def sq(x): return x**2`
- **闭包**：内层函数记住外层变量，C/Java没有
- **装饰器**：`@timer` 在不修改原函数的情况下增加功能
- **生成器**：`yield` 逐个产生值，不一次性占内存

**动手练习：**
1. 给 `calculate_avg_degree` 函数添加类型注解，然后用 `help()` 查看
2. 修改 `retry` 装饰器的 `max_attempts=5`，运行看效果
3. 把 `generate_edges` 生成器改成普通函数（用 `return edges`），体会内存差异

---

## Day 4：文件读写

```bash
python src/phase1_fundamentals/05_file_io.py
```

**Python读写的几个层次：**
1. `open()/read()/write()` —— 原始文本
2. `csv.DictReader/DictWriter` —— 表格数据
3. `json.dump/json.load` —— 结构化数据（网络数据常用！）
4. `pickle.dump/pickle.load` —— Python对象序列化

**网络分析中最常用的数据格式：**
- **边列表** (Edge List)：逐行 `u v`，简单粗暴
- **CSV**：带权重的边 `source,target,weight`
- **JSON**：完整的网络描述 `{"nodes": [...], "edges": [...]}`
- **GML**：NetworkX原生格式

**动手练习：**
1. 打开生成的 `data/edges.csv` 和 `data/network.json`，看看实际长什么样
2. 用记事本创建一个边列表文件，然后用Python读取并构建邻接表

---

## Day 5：面向对象

```bash
python src/phase1_fundamentals/06_oop.py
```

**Python OOP vs Java OOP：**

| 特性 | Java | Python |
|------|------|--------|
| 构造函数 | `ClassName()` | `__init__(self)` |
| this | `this.x` | `self.x` |
| 继承 | `extends` | `class Child(Parent)` |
| 接口 | `interface` | 鸭子类型（不需要） |
| toString | `toString()` | `__repr__()` / `__str__()` |
| getter/setter | 手写 | `@property` 装饰器 |
| 简化类 | Lombok | `@dataclass` |

**这个文件展示的设计模式——也是整个项目的核心架构：**
```
Network (基类)
  ├── RandomNetwork (ER随机网络)
  └── ScaleFreeNetwork (BA无标度网络)
```

**动手练习：**
1. 仿照 `ScaleFreeNetwork`，自己实现一个 `SmallWorldNetwork` 类
2. 把 `exercise_class_hierarchy()` 中的 `ErdosRenyi` 构造参数 `p=0.02` 改成 `0.1`，观察边数变化

---

## Day 6：高级特性

```bash
python src/phase1_fundamentals/07_advanced_features.py
```

**这些你面试时一定会被问到：**
- **列表/字典/集合推导式** —— 一行代码解决问题
- **生成器管道** —— 处理大数据不爆内存
- **装饰器** —— `@retry` 自动重试、`@timer` 自动计时
- **上下文管理器** —— `with` 语句自动清理资源
- **`itertools`** —— 排列组合、无限迭代器

**动手练习：**
1. 理解 "生成器是惰性求值" —— 对比 `list(range(10000000))` vs `range(10000000)` 的内存占用
2. 试试 `itertools.combinations(G.nodes(), 3)` —— 这能枚举网络中所有可能的三角形

---

## Phase 1 学完的检验标准

在终端中输入：
```bash
python -c "
from collections import Counter, defaultdict
# 1. 用列表推导式生成100个偶数
evens = [x for x in range(200) if x % 2 == 0]
print(f'1. 列表推导式: {len(evens)}个偶数')

# 2. 用Counter统计
d = Counter([1,2,2,3,3,3,4,4,4,4])
print(f'2. Counter: {d}')

# 3. 用defaultdict
adj = defaultdict(list)
adj[0].append(1)
print(f'3. defaultdict: {dict(adj)}')

# 4. 写一个lambda
sq = lambda x: x**2
print(f'4. lambda: {sq(5)}')

# 5. 写一个dataclass
from dataclasses import dataclass
@dataclass
class Node:
    id: int
    label: str = ''

n = Node(id=1, label='hub')
print(f'5. dataclass: {n}')
"
```
如果以上5段代码你都能看懂并写出，Phase 1就掌握了。

---

# Phase 2：NetworkX + 网络基本概念（预计3-4天）

> **目标**：学会用NetworkX创建和分析网络
> **对应导师研究**：网络结构的数学描述（PPT第2-3章）

---

## Day 1-2：网络的基本度量

```bash
python src/phase2_network_basics/01_networkx_basics.py
```

**这个文件教你的核心概念（每个都对应导师PPT里的一章）：**

| 概念 | 含义 | 函数 | PPT章节 |
|------|------|------|---------|
| **度 (Degree)** | 节点有多少邻居 | `G.degree()` | PPT 3 |
| **集聚系数** | 朋友的朋友是不是朋友 | `nx.clustering(G)` | PPT 3 |
| **最短路径** | 两节点最少经过几条边 | `nx.shortest_path(G)` | PPT 3 |
| **介数** | 经过该节点的最短路径数 | `nx.betweenness_centrality(G)` | PPT 3 |
| **邻接矩阵** | 网络的矩阵表示 | `nx.adjacency_matrix(G)` | PPT 2 |
| **拉普拉斯矩阵** | 邻接矩阵的变体 | `nx.laplacian_matrix(G)` | PPT 2 |

**动手练习（一定要做！）：**
1. 修改节点数和边的连接方式，观察平均集聚系数怎么变化
2. 把一个节点度改成很大，它的介数中心性会怎么变？
3. 运行末尾的 `exercise_full_analysis()`，读懂每一项输出的含义
4. 用NetworkX生成一个500节点的BA网络，对比它和100节点BA网络的差别

**理解这些概念对导师课题的意义：**
- **度分布** → 判断网络是否无标度（幂律）
- **集聚系数** → 判断网络是否有小世界特性
- **介数** → 找出关键节点（级联失效的攻击目标）
- **邻接矩阵** → 所有计算的基础

---

## Day 3-4：自己动手写分析函数

```python
# 在Python交互环境中试试：

import networkx as nx
import numpy as np
from collections import Counter

# 步骤1：生成BA网络
ba = nx.barabasi_albert_graph(1000, 3, seed=42)

# 步骤2：计算度分布
degrees = [d for _, d in ba.degree()]
dist = Counter(degrees)

# 步骤3：在log-log坐标下看是否是直线（幂律）
import matplotlib.pyplot as plt
k_vals = sorted(dist.keys())
p_vals = [dist[k]/1000 for k in k_vals]
plt.scatter(k_vals, p_vals)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.show()
```

如果log-log图看起来像一条直线→这个网络服从幂律分布→无标度网络！

---

# Phase 3：四种经典网络模型（预计3-4天）

> **目标**：理解并实现四种网络模型，理解它们的本质差异
> **对应导师研究**：PPT第4章

---

## Day 1-2：运行对比实验

```bash
python src/phase3_network_models/01_four_models.py
```

**四种模型的直观理解：**

| 模型 | 现实例子 | 度分布形状 | 核心机制 |
|------|----------|-----------|----------|
| **规则网络** | 晶体结构 | Delta函数（所有度相同） | 最近邻连接 |
| **ER随机** | 随机认识陌生人 | Poisson（钟形曲线） | 随机连接 |
| **WS小世界** | 社交网络 | 接近Poisson | 随机重连 |
| **BA无标度** | 互联网、论文引用 | 幂律（长尾） | 增长+偏好连接 |

**看结果图（results目录下已生成）：**
- `degree_distribution.png`：四张子图，左下角BA的log-log图应该接近直线
- `network_visualization.png`：BA图里能看到少数大节点（hub）和大量小节点

**控制台输出要关注：**
```
BA无标度:
  平均度: 5.96
  最大度: 66       # ← 注意！最大度是平均度的11倍
  集聚系数: 0.0559
  平均路径长度: 3.19  # ← 短路径，小世界效应
```

```
BA 幂律拟合:
  幂律指数 gamma = 1.77    # ← 接近2，典型的无标度网络
  R-squared = 0.8235       # ← 拟合很好
```

---

## Day 3-4：动手实验

1. **修改参数看变化：**
   - 把BA的 `m=3` 改成 `m=5`，最大度怎么变？
   - 把ER的 `p=0.05` 改成 `p=0.001`，网络还连通吗？
   - 把WS的 `p=0.3` 改成 `p=0.01`，集聚系数怎么变？

2. **回答这些问题（写到笔记里）：**
   - 为什么BA网络的平均距离很小？（因为hub节点充当"桥梁"）
   - 为什么真实网络大多是BA型而非ER型？（增长+偏好连接无处不在）
   - 规则网络和WS小世界网络的根本区别是什么？（连边方式）

---

# Phase 4：网络鲁棒性分析（预计4-5天）

> **目标**：模拟攻击网络，理解无标度网络的"阿喀琉斯之踵"
> **对应导师研究**：PPT第6章，Albert et al. (2000) Nature

---

## Day 1-2：运行鲁棒性实验

```bash
python src/phase4_robustness/01_robustness_analysis.py
```

**这个实验是导师研究的基石！**

实验做三件事：
1. **随机故障**：随机删除节点
2. **蓄意攻击（度）**：按度从高到低删除
3. **蓄意攻击（介数）**：按介数从高到低删除

**看结果图（`er_vs_ba_robustness.png`）：**
- 蓝实线（BA+随机故障）：接近水平 → BA对随机故障鲁棒
- 红实线（BA+蓄意攻击）：快速下降 → BA对蓄意攻击极其脆弱
- 蓝虚线（ER+随机故障）和红虚线（ER+蓄意攻击）：差不多

**这验证了导师PPT里的核心结论：**
> 无标度网络对随机故障鲁棒，但对蓄意攻击极其脆弱

---

## Day 2-3：理解Malloy-Reed判据

控制台会输出：
```
Malloy-Reed判据验证:
  <k> = 5.96
  <k^2> = 82.52
  kappa = <k^2>/<k> = 13.84  # ← > 2，巨连通分量存在
```

**判据的物理含义：**
- κ > 2：网络存在巨连通分量（大部分节点连通）
- κ < 2：网络碎片化（没有巨连通分量）

为什么BA的κ很大？因为存在度很高的hub节点 → `<k²>` 很大 → κ很大 → 对随机故障很鲁棒

**动手实验：**
1. 修改 `targeted_removal_by_degree` 只移除度最大的1个节点，观察连通性变化
2. 创建一个500节点的ER网络和BA网络，对比它们的 `κ` 值
3. 试试 `targeted_removal_by_betweenness(G, 0.05)`，和按度攻击对比效果

---

## Day 4-5：理解渗流相变

**二级相变（ER网络）：**
- 巨连通分量**连续、缓慢**地缩小 → 可以预测

**一级相变（相依网络，Phase 6会讲）：**
- 巨连通分量**突然、跳跃式**崩溃 → 无法预警

修改代码，把 `fractions` 的粒度调细（比如200个点），再画一次曲线，观察临界点的形状。

---

# Phase 5：级联失效仿真（预计5-7天）

> **目标**：实现负载-容量模型，模拟过载级联失效
> **对应导师研究**：Duan et al. 负载最近邻分配模型
> **这直接对应导师的核心方向！**

---

## Day 1-2：理解负载-容量模型

```bash
python src/phase5_cascading/01_cascading_failure.py
```

### 模型的核心逻辑（一行一行理解）：

```
1. 每个节点有初始负载 L_i （通常与度或介数成正比）
2. 每个节点有容量上限 C_i = (1+α) × L_i
3. 攻击某个节点 → 它失效 → 负载重新分配给邻居
4. 邻居收到额外负载 → 如果超过容量 → 邻居也失效
5. 级联继续 → 直到没有新节点过载
```

### 三个分配策略（这是导师论文的核心贡献！）：

| 策略 | 含义 | `load_strategy` |
|------|------|----------------|
| **按度分配** | 高度节点分更多 | `"degree"` |
| **平均分配** | 所有邻居平分 | `"equal"` |
| **最近邻分配** | 优先分给低度节点 | `"nearest_neighbor"` |

---

## Day 3-4：运行实验并理解结果

**实验1：容量参数α的影响（看 `cascading_alpha.png`）**

α越小 → 容量越紧 → 更容易过载 → 级联越大
α越大 → 容量越宽松 → 更不容易过载 → 级联越小

**实验2：可调参数β的影响（看 `cascading_beta.png`）**

β > 0：分配向高度节点倾斜
β < 0：分配向低度节点倾斜
β = 0：平均分配

这是导师的**可调负载分配模型**的核心概念！

---

## Day 5-7：动手深度实验

1. **攻击不同节点：**
```python
# 攻击hub节点 vs 攻击叶子节点
sim = CascadingFailureSimulator(ba, alpha=0.3)
_, failed_hub, _ = sim.simulate_single_attack(hub_id)    # hub节点
_, failed_leaf, _ = sim.simulate_single_attack(leaf_id)   # 叶子节点
# 对比两者的差异！
```

2. **修改负载初始化逻辑：**
   - 当前：`loads[node] = G.degree(node)`（用度做负载）
   - 改成：`loads[node] = betweenness[node] * 100`（用介数做负载）
   - 观察结果有什么不同？

3. **实现自己的分配策略：**
```python
# 在 _redistribute_load 中添加新策略：
elif self.load_strategy == "my_strategy":
    # 你的自定义逻辑
    pass
```

---

# Phase 6：相依网络级联失效（预计5-7天）

> **目标**：模拟双层耦合网络的级联崩溃，复现导师PNAS 2019论文
> **对应导师研究**：Duan et al., PNAS 2019
> **这是导师最重要的论文！**

---

## Day 1-2：理解相依网络

```bash
python src/phase6_interdependent/01_interdependent_cascading.py
```

### 相依网络的概念（导师PPT第7.3章）：

想象两个网络——电力网和互联网：
- 电站需要互联网来控制 → 互联网节点失效 → 电站无法控制 → 电站失效
- 互联网需要电站供电 → 电站失效 → 互联网节点断电 → 互联网节点失效
- 两者互相依赖 → 失效在两个网络间交替传播 → **级联崩溃**

### 代码中的级联过程：

```
Step 1: Layer A被攻击，部分节点失效
Step 2: Layer A的失效节点在Layer B的依赖节点也失效
Step 3: Layer B失效节点的邻居（在Layer A有依赖关系）也失效
Step 4: 检查断裂 → 不在巨连通分量的节点视为失效
Step 5: 重复直到稳定
```

---

## Day 3-4：理解一级相变

**结果图 `interdependent_percolation.png`：**
- X轴：移除比例 f
- Y轴：存活比例

关键观察：曲线不是在某个点平滑下降到零，而是**在某个临界点突然跳变到零**！

- 二级相变：平滑下降（单层网络的渗流）
- **一级相变：突然崩溃（相依网络的特征）** ← 导师PNAS论文的核心发现

**一级相变更危险的原因：**
- 没有预警信号
- 系统看似稳定，但超过临界点后瞬间崩溃

---

## Day 5-7：动手实验

1. **改变耦合方式：**
```python
# 一一对应耦合
net1 = InterdependentNetwork(GA, GB, coupling_type="one_to_one")

# 随机耦合
net2 = InterdependentNetwork(GA, GB, coupling_type="random")

# 对比两种耦合方式下的相变行为
```

2. **改变耦合密度：**
   - `coupling_strength.png` 展示了耦合比例对级联的影响
   - 耦合比例越高 → 网络越脆弱

3. **替换网络模型：**
   - 当前Layer A和B都是BA网络
   - 试试把Layer A换成ER，Layer B也是ER
   - 试试Layer A=BA, Layer B=ER（非对称耦合）

4. **思考延伸（对应导师PNAS论文）：**
   - 如果有三层网络怎么办？
   - 如果耦合关系不是一对一的怎么办？
   - 如果考虑节点负载和容量（结合Phase 5）怎么办？

---

# Phase 7：可视化与工程化（预计3-4天）

> **目标**：制作专业级别的网络分析图表

---

## Day 1-2：运行可视化

```bash
python src/phase7_visualization/01_network_viz.py
```

### 生成的四类图表：

1. **基础网络图**（`ba_network.png`, `er_network.png`）
   - 节点大小与度成正比
   - 颜色深浅反映度数

2. **综合分析仪表板**（`ba_dashboard.png`, `er_dashboard.png`）
   - 六合一：结构+度分布+直方图+集聚系数+介数+统计信息

3. **级联过程动画**（函数 `visualize_cascade_process`）
   - 每一步展示哪些节点新失效（红色）

4. **度分布图**——已在Phase 3中生成

---

## Day 3-4：定制自己的图表

1. **修改配色方案：**
```python
# 在代码中找到 cmap=plt.cm.YlOrRd，改成：
cmap=plt.cm.Blues     # 蓝色调
cmap=plt.cm.viridis    # 默认的科学配色
```

2. **调整布局：**
```python
pos = nx.spring_layout(G, seed=42, k=0.3)   # 调整k值改变节点间距
pos = nx.circular_layout(G)                   # 换成圆形布局
pos = nx.kamada_kawai_layout(G)              # 力导向布局
```

3. **保存高清图：**
```python
plt.savefig("output.png", dpi=300, bbox_inches="tight")  # dpi=300是论文要求
```

---

# Phase 8：机器学习预测节点重要性（预计3-5天）

> **目标**：用ML模型预测网络中的关键节点，完成"数据驱动的网络科学"闭环
> **对应就业**：简历上的ML项目 + 面试中的sklearn八股文

---

## Day 1-2：理解特征工程

```bash
python src/phase8_ml_integration/01_ml_prediction.py
```

**这个模块做三件事：**
1. **特征提取**：从网络中提取每个节点的8个特征（度、聚类系数、中心性、PageRank等）
2. **标签生成**：用度/介数排名标记"重要节点"
3. **模型训练**：对比随机森林、梯度提升、逻辑回归三种模型

**核心概念：**

| 概念 | 含义 | 对应面试问题 |
|------|------|-------------|
| 特征工程 | 从原始数据中提取有用信息 | "你的模型用了哪些特征？" |
| 训练/测试集划分 | 用一部分数据训练，另一部分验证 | "为什么要划分数据集？" |
| 特征标准化 | 把不同量纲的特征缩放到同一范围 | "什么时候需要标准化？" |
| 交叉验证 | 多次划分数据集取平均 | "交叉验证的作用？" |
| 过拟合 | 模型在训练集上好但测试集上差 | "怎么判断和解决过拟合？" |

## Day 3-5：动手实验

1. **修改标签策略**：把 `label_important_nodes` 的 `top_fraction=0.1` 改成 `0.05` 或 `0.2`，观察模型性能变化
2. **添加新特征**：在 `extract_node_features` 中加入三角形数量、平均邻居度等特征
3. **跨网络测试**：在ER和WS网络上训练，看看在BA网络上效果如何（泛化能力）
4. **调参**：修改随机森林的 `n_estimators` 和 `max_depth`，观察性能变化

---

整个项目学完后的能力对照

### Python工程能力
- [ ] 变量、类型、控制流 —— 和C/Java的差异
- [ ] 列表/字典/集合/元组 —— 切片、推导式
- [ ] 函数/lambda/闭包/装饰器/生成器
- [ ] 面向对象 —— 继承、多态、@property、@dataclass
- [ ] 文件读写 —— CSV/JSON/Pickle
- [ ] 异常处理、上下文管理器
- [ ] itertools、functools等标准库

### 科学计算能力
- [ ] NumPy数组操作
- [ ] SciPy拟合与统计
- [ ] Matplotlib专业图表
- [ ] NetworkX网络分析

### 网络科学专业能力
- [ ] 四种网络模型的本质差异
- [ ] 度分布、集聚系数、介数的含义和计算
- [ ] 渗流相变 — 二级相变 vs 一级相变
- [ ] Malloy-Reed判据
- [ ] 负载-容量级联失效模型
- [ ] 可调负载分配策略
- [ ] 相依网络的级联崩溃
- [ ] 无标度网络的"阿喀琉斯之踵"

### 导师课题对应
- [ ] 网络鲁棒性（Phase 4）← Albert et al. (2000)
- [ ] 过载级联失效（Phase 5）← 导师负载分配模型
- [ ] 相依网络级联失效（Phase 6）← 导师PNAS 2019

### 机器学习能力
- [ ] sklearn分类器（随机森林、梯度提升、逻辑回归）
- [ ] 特征工程（从网络中提取节点特征）
- [ ] 模型评估（准确率、精确率、召回率、F1、交叉验证）
- [ ] 特征重要性分析
- [ ] 数据集划分与标准化

---

## 学习节奏建议

```
时间          |  内容
--------------|-------------------------------------
第1周         | Phase 1-2（Python基础 + NetworkX入门）
第2-3周       | Phase 3-4（网络模型 + 鲁棒性分析）
第4-5周       | Phase 5（级联失效仿真——这是核心）
第6-7周       | Phase 6（相依网络——导师方向的核心）
第8周         | Phase 7（可视化）+ Phase 8（ML预测）
第9周         | 整理笔记、上传GitHub、写README
```

每个Phase结束后，尝试用自己的话给导师讲一遍你学到了什么——这是验证你是否真正理解的最佳方式。

---

## 常见问题

**Q: 代码报错了怎么办？**
直接复制报错信息给我，我帮你调。

**Q: 某个概念不懂？**
直接问，我可以用简单例子解释。

**Q: 想跳过Phase 1直接从Phase 2开始？**
可以，但你可能会在Phase 5-6的函数装饰器和OOP卡住。

**Q: 学完了想进阶？**
Phase 8 已经写好了！运行 `python src/phase8_ml_integration/01_ml_prediction.py` 即可体验。
进阶方向：用GNN（图神经网络）替代传统ML做节点重要性预测。

**Q: 想把这个项目写到简历上？**
完全可以。描述："独立开发了复杂网络级联失效仿真平台，实现了负载-容量模型、可调负载分配策略和双层相依网络级联模拟，复现了Duan et al. PNAS 2019的核心结果。并用随机森林/梯度提升预测网络关键节点（F1=0.93）。"

---

## 附录：pytest 测试命令速查

### 前提：激活虚拟环境

每次打开 VS Code 终端，先确认左下角显示 `.venv (Python 3.9.13)`。
如果没有，在终端里手动激活：

```powershell
# PowerShell（VS Code 默认终端）
& ".venv\Scripts\Activate.ps1"

# Git Bash
source .venv/Scripts/activate
```

### 运行测试

```bash
# 运行全部测试（详细输出）
python -m pytest test_network.py -v

# 运行全部测试（简略输出）
python -m pytest test_network.py

# 只运行某一个测试函数
python -m pytest test_network.py -v -k "test_er_network_avg_degree"

# 运行包含关键词的测试
python -m pytest test_network.py -v -k "ba"

# 第一个失败就停止（不用跑完全部）
python -m pytest test_network.py -x

# 显示失败详情 + print 输出
python -m pytest test_network.py -v -s

# 只收集测试列表，不实际运行（看看有哪些测试）
python -m pytest test_network.py --collect-only
```

### 为什么用 `python -m pytest` 而不是 `pytest`？

| 命令 | 说明 |
|------|------|
| `pytest xxx` | 直接调用 pytest 可执行文件，路径写死在安装时，项目移动后可能报错 |
| `python -m pytest xxx` | 用当前 Python 解释器启动 pytest，**永远不出错** |

本项目的 `.venv` 是从旧路径迁移过来的，`pytest.exe` 里的路径已经失效，所以**必须用 `python -m pytest`**。

### 测试文件在哪个目录运行？

```bash
# 在项目根目录运行（推荐）
cd E:\学习论文（研究生期间）\学业规划\ai_python
python -m pytest tests/test_network.py -v

# 或者在 tests 目录内运行
cd E:\学习论文（研究生期间）\学业规划\ai_python\tests
python -m pytest test_network.py -v
```

注意路径变化：在根目录用 `tests/test_network.py`，在 tests 目录用 `test_network.py`。

### 测试文件里的函数命名规则

pytest 会自动收集以下命名的函数：
- 函数名以 `test_` 开头，如 `test_er_network_avg_degree`
- 文件名以 `test_` 开头，如 `test_network.py`

不是这种命名的函数会被忽略，比如 `_import_from()` 只是辅助函数，不会被当作测试。

### 查看测试覆盖率（可选进阶）

```bash
# 先安装 coverage
python -m pip install pytest-cov

# 运行并生成覆盖率报告
python -m pytest test_network.py -v --cov=src --cov-report=term-missing
```

这会告诉你 src/ 下的代码有多少被测试覆盖了，哪些行还没被测试到。
