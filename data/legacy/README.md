# data/legacy · 遗留数据集

这里放**不属于研究内容 1、但删了可惜**的数据。只读，不再更新。

## node_importance_dataset.csv（396 KB，6000 行 × 14 列）

节点重要度数据集：**30 个网络 × 200 节点**（BA / ER / WS 各 2000 行）。

| 列 | 含义 |
|---|---|
| `node_id` / `network_id` / `model_type` | 节点、网络编号、网络模型（BA / ER / WS） |
| `degree` `clustering_coeff` `degree_centrality` `betweenness_centrality` `closeness_centrality` `k_core` `pagerank` | 7 个静态拓扑特征 |
| `induced_failures` | 单点触发级联后**除自身外**倒下的节点数（10.95% 非零） |
| `gcc_drop` | 巨连通分量跌幅 |
| `sir_influence` | SIR 传播影响范围（100% 非零） |
| `is_important` | 标签（top 15%） |

### 来源与语义（重要）

- 生成自**已删除**的项目「GNN关键节点识别实战 —— 从sklearn到FINDER」；
  其生成函数是「复杂网络韧性分析工具 —— 研0暑假实战项目」里
  `resilience.py::CascadeFailureSimulator.label_by_cascade_consequence(top_fraction=0.15)`：
  **对每个节点单独触发一次级联**，把"连带倒下的节点数"作为真实重要度，再取 top 15% 作标签
  （目的是修掉"标签=度、特征里也有度"的**标签循环论证**）。
- 级联模型是 **Motter–Lai 负载-容量**（单层），对应本平台 `phase5_cascading`，**不是** phase6 的相依渗流。
- **标签循环论证的教训（实测）**：用"度 top15%"当标签时，度中心性预测 F1 虚高；改用级联后果后该捷径被堵住。
- **主标签选 `sir_influence`**：`induced_failures` 只有 10.95% 非零（太稀疏，正类不足），`sir_influence` 100% 非零。

### 为什么留在这里

GNN 关键节点识别方向已被导师叫停（"已有多人负责"），项目本体已删；
但这份数据是唯一不可再生的产出，且与"**脆弱性探针**：单点级联后果 I_i"这一口径直接相关
（见 `src/metrics/resilience_metrics.py::vulnerability_probe`），所以留档备查。

**如需重造**：本项目已不含生成器。用 `phase5_cascading` 的 Motter–Lai 级联 +
上文的标签规则（单点触发 → 除自身外的连带失效数 → top 15%）即可等价复现，注意固定种子。
