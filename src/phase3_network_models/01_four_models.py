"""
Phase 3.1 - 四种经典网络模型
实现并对比：规则网络、ER随机网络、WS小世界网络、BA无标度网络
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 规则网络（Regular Network）
# ============================================================

def create_regular_network(n, k):
    """
    创建规则网络（最近邻连接）
    n: 节点数
    k: 每个节点连接的邻居数（必须为偶数）
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            G.add_edge(i, neighbor)

    return G

# ============================================================
# 2. ER随机网络（已由NetworkX内置）
# ============================================================

# nx.erdos_renyi_graph(n, p, seed)

# ============================================================
# 3. WS小世界网络（已由NetworkX内置）
# ============================================================

# nx.watts_strogatz_graph(n, k, p, seed)

# ============================================================
# 4. BA无标度网络（已由NetworkX内置）
# ============================================================

# nx.barabasi_albert_graph(n, m, seed)

# ============================================================
# 5. 对比四种模型的统计特性
# ============================================================

def compare_models(n=1000, k=6, m=3, p=0.05):
    """对比四种网络模型"""
    print(f"网络规模: N={n}")
    print(f"ER: p={p}")
    print(f"WS: K={k}, p={p}")
    print(f"BA: m={m}")
    print("=" * 70)

    # 创建网络
    regular = create_regular_network(n, k)
    er = nx.erdos_renyi_graph(n, p, seed=42)
    ws = nx.watts_strogatz_graph(n, k, p, seed=42)
    ba = nx.barabasi_albert_graph(n, m, seed=42)

    models = {
        "规则网络": regular,
        "ER随机": er,
        "WS小世界": ws,
        "BA无标度": ba
    }

    results = {}
    for name, G in models.items():
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees)
        max_degree = max(degrees)
        clustering = nx.average_clustering(G)
        density = nx.density(G)

        # 平均路径长度（需要连通图）
        if nx.is_connected(G):
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # 取最大连通分量
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)

        results[name] = {
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "clustering": clustering,
            "density": density,
            "avg_path": avg_path,
            "diameter": diameter
        }

        print(f"\n{name}:")
        print(f"  平均度: {avg_degree:.2f}")
        print(f"  最大度: {max_degree}")
        print(f"  集聚系数: {clustering:.4f}")
        print(f"  密度: {density:.6f}")
        print(f"  平均路径长度: {avg_path:.2f}")
        print(f"  直径: {diameter}")

    return models, results

# ============================================================
# 6. 度分布对比
# ============================================================

def plot_degree_distribution(models, save_path=None):
    """绘制四种模型的度分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, G) in zip(axes.flatten(), models.items()):
        degrees = [d for _, d in G.degree()]
        degree_dist = Counter(degrees)

        k_vals = sorted(degree_dist.keys())
        p_vals = [degree_dist[k] / len(degrees) for k in k_vals]

        ax.scatter(k_vals, p_vals, s=20, alpha=0.7)
        ax.set_xlabel("度 k")
        ax.set_ylabel("P(k)")
        ax.set_title(f"{name} - 度分布")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n度分布图已保存: {save_path}")
    plt.close()

# ============================================================
# 7. 幂律分布检验（BA网络应该服从幂律）
# ============================================================

def check_power_law(G, name):
    """检验度分布是否服从幂律"""
    from scipy import stats

    degrees = [d for _, d in G.degree() if d > 0]  # 排除度为0的节点
    degree_dist = Counter(degrees)

    k_vals = np.array(sorted(degree_dist.keys()))
    p_vals = np.array([degree_dist[k] for k in k_vals])

    # 对数变换
    log_k = np.log(k_vals.astype(float))
    log_p = np.log(p_vals / sum(p_vals))

    # 过滤掉无效值
    valid = np.isfinite(log_k) & np.isfinite(log_p)
    log_k, log_p = log_k[valid], log_p[valid]

    if len(log_k) < 3:
        print(f"\n{name}: 数据点不足，跳过幂律拟合")
        return None, None

    # 线性回归拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)

    gamma = -slope  # 幂律指数
    r_squared = r_value ** 2

    print(f"\n{name} 幂律拟合:")
    print(f"  幂律指数 gamma = {gamma:.2f}")
    print(f"  R-squared = {r_squared:.4f}")

    if r_squared > 0.8 and gamma > 1:
        print(f"  结论: 符合幂律分布")
    else:
        print(f"  结论: 不符合幂律分布")

    return gamma, r_squared

# ============================================================
# 8. 网络可视化
# ============================================================

def visualize_networks(models, save_path=None):
    """可视化四种网络的结构"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (name, G) in zip(axes.flatten(), models.items()):
        # 使用spring layout
        pos = nx.spring_layout(G, seed=42, k=0.3)

        # 节点大小与度成正比
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 10 + 20 for node in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color="steelblue", alpha=0.8)
        ax.set_title(f"{name} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"网络可视化图已保存: {save_path}")
    plt.close()

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 对比四种模型
    models, results = compare_models(n=500, k=6, m=3, p=0.05)

    # 绘制度分布
    plot_degree_distribution(models, "E:/ai_python/results/degree_distribution.png")

    # 幂律检验
    print("\n" + "="*50)
    print("幂律分布检验")
    print("="*50)
    for name, G in models.items():
        check_power_law(G, name)

    # 可视化
    visualize_networks(models, "E:/ai_python/results/network_visualization.png")

    print("\n完成！所有图表已保存到 results/ 目录")
