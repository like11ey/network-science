"""
Phase 2.1 - NetworkX基础
NetworkX是Python最强大的网络分析库
"""

import networkx as nx
import numpy as np
from collections import Counter
from pathlib import Path

# 项目数据目录
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 创建网络
# ============================================================

# 创建空图
G = nx.Graph()           # 无向图
D = nx.DiGraph()         # 有向图
W = nx.Graph()           # 加权图

# 添加节点
G.add_node(0)
G.add_nodes_from([1, 2, 3, 4, 5])

# 添加边
G.add_edge(0, 1)
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (0, 2)])

print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
print(f"节点列表: {list(G.nodes())}")
print(f"边列表: {list(G.edges())}")

# ============================================================
# 2. 网络基本属性
# ============================================================

print(f"\n基本属性:")
print(f"  是否有向: {G.is_directed()}")
print(f"  是否加权: {nx.is_weighted(G)}")
print(f"  是否连通: {nx.is_connected(G)}")
print(f"  连通分量数: {nx.number_connected_components(G)}")

# 节点和边的属性
G.nodes[0]["label"] = "Hub节点"
G[0][1]["weight"] = 2.5
print(f"\n节点0的属性: {dict(G.nodes[0])}")
print(f"边(0,1)的属性: {dict(G[0][1])}")

# ============================================================
# 3. 度（Degree）
# ============================================================

print(f"\n度分析:")
# 每个节点的度
for node in G.nodes():
    print(f"  节点{node}: 度={G.degree(node)}")

# 度序列
degrees = [d for _, d in G.degree()]
print(f"度序列: {degrees}")
print(f"最大度: {max(degrees)}")
print(f"最小度: {min(degrees)}")
print(f"平均度: {np.mean(degrees):.2f}")

# 度分布
degree_dist = Counter(degrees)
print(f"度分布: {dict(sorted(degree_dist.items()))}")

# ============================================================
# 4. 邻居
# ============================================================

print(f"\n邻居分析:")
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    print(f"  节点{node}的邻居: {neighbors}")

# 两个节点是否相连
print(f"\n  节点0和1是否相连: {G.has_edge(0, 1)}")
print(f"  节点0和5是否相连: {G.has_edge(0, 5)}")

# ============================================================
# 5. 路径和距离
# ============================================================

print(f"\n路径分析:")
# 最短路径
if nx.is_connected(G):
    shortest_path = nx.shortest_path(G, 0, 5)
    print(f"  从0到5的最短路径: {shortest_path}")
    print(f"  最短路径长度: {nx.shortest_path_length(G, 0, 5)}")

    # 所有节点对的最短路径长度
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"  平均最短路径长度: {avg_path_length:.2f}")

    # 直径（最大最短路径）
    diameter = nx.diameter(G)
    print(f"  网络直径: {diameter}")

# ============================================================
# 6. 集聚系数（Clustering Coefficient）
# ============================================================

print(f"\n集聚系数:")
# 每个节点的集聚系数
for node in G.nodes():
    cc = nx.clustering(G, node)
    print(f"  节点{node}: 集聚系数={cc:.3f}")

# 网络的平均集聚系数
avg_clustering = nx.average_clustering(G)
print(f"  平均集聚系数: {avg_clustering:.3f}")

# ============================================================
# 7. 介数中心性（Betweenness Centrality）
# ============================================================

print(f"\n介数中心性:")
betweenness = nx.betweenness_centrality(G)
for node, bc in sorted(betweenness.items(), key=lambda x: -x[1]):
    print(f"  节点{node}: 介数={bc:.3f}")

# 度中心性
degree_centrality = nx.degree_centrality(G)
print(f"\n度中心性:")
for node, dc in sorted(degree_centrality.items(), key=lambda x: -x[1]):
    print(f"  节点{node}: 度中心性={dc:.3f}")

# ============================================================
# 8. 邻接矩阵
# ============================================================

print(f"\n邻接矩阵:")
adj_matrix = nx.adjacency_matrix(G)
print(f"类型: {type(adj_matrix)}")
print(f"形状: {adj_matrix.shape}")
print(f"稠密矩阵:\n{adj_matrix.toarray()}")

# 转为numpy数组
adj_array = nx.to_numpy_array(G)
print(f"\nNumPy数组:\n{adj_array}")

# ============================================================
# 9. 拉普拉斯矩阵
# ============================================================

print(f"\n拉普拉斯矩阵:")
laplacian = nx.laplacian_matrix(G)
print(f"稠密形式:\n{laplacian.toarray()}")

# 拉普拉斯矩阵的特征值
eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
print(f"特征值: {np.sort(eigenvalues)}")

# ============================================================
# 10. 网络生成（NetworkX内置）
# ============================================================

print(f"\n--- 使用NetworkX内置生成器 ---")

# ER随机网络
er = nx.erdos_renyi_graph(100, 0.05, seed=42)
print(f"\nER网络: N=100, p=0.05")
print(f"  节点数: {er.number_of_nodes()}")
print(f"  边数: {er.number_of_edges()}")
print(f"  平均度: {2 * er.number_of_edges() / er.number_of_nodes():.2f}")
print(f"  是否连通: {nx.is_connected(er)}")
if nx.is_connected(er):
    print(f"  平均路径长度: {nx.average_shortest_path_length(er):.2f}")
print(f"  平均集聚系数: {nx.average_clustering(er):.4f}")

# WS小世界网络
ws = nx.watts_strogatz_graph(100, 6, 0.3, seed=42)
print(f"\nWS网络: N=100, K=6, p=0.3")
print(f"  平均度: {2 * ws.number_of_edges() / ws.number_of_nodes():.2f}")
print(f"  平均路径长度: {nx.average_shortest_path_length(ws):.2f}")
print(f"  平均集聚系数: {nx.average_clustering(ws):.4f}")

# BA无标度网络
ba = nx.barabasi_albert_graph(100, 3, seed=42)
print(f"\nBA网络: N=100, m=3")
print(f"  平均度: {2 * ba.number_of_edges() / ba.number_of_nodes():.2f}")
print(f"  平均路径长度: {nx.average_shortest_path_length(ba):.2f}")
print(f"  平均集聚系数: {nx.average_clustering(ba):.4f}")

# ============================================================
# 11. 导出网络
# ============================================================

print(f"\n导出网络:")
# 保存为GML格式
nx.write_gml(ba, str(DATA_DIR / "ba_network.gml"))

# 保存为边列表
nx.write_edgelist(ba, str(DATA_DIR / "ba_edges.txt"))

# 保存为邻接表
nx.write_adjlist(ba, str(DATA_DIR / "ba_adjlist.txt"))

print("  网络已保存为多种格式")

# ============================================================
# 练习：完整分析一个网络
# ============================================================

def exercise_full_analysis():
    """完整分析一个BA无标度网络"""
    print("\n" + "="*50)
    print("BA无标度网络完整分析")
    print("="*50)

    # 创建网络
    G = nx.barabasi_albert_graph(500, 3, seed=42)

    # 基本信息
    print(f"\n1. 基本信息:")
    print(f"   节点数: {G.number_of_nodes()}")
    print(f"   边数: {G.number_of_edges()}")
    print(f"   密度: {nx.density(G):.4f}")
    print(f"   是否连通: {nx.is_connected(G)}")

    # 度分析
    degrees = [d for _, d in G.degree()]
    print(f"\n2. 度分析:")
    print(f"   平均度: {np.mean(degrees):.2f}")
    print(f"   最大度: {max(degrees)}")
    print(f"   最小度: {min(degrees)}")
    print(f"   度标准差: {np.std(degrees):.2f}")

    # 度分布
    degree_dist = Counter(degrees)
    print(f"\n3. 度分布:")
    for deg, freq in sorted(degree_dist.items())[:10]:
        print(f"   度={deg}: {freq}次")

    # 路径分析
    print(f"\n4. 路径分析:")
    print(f"   平均路径长度: {nx.average_shortest_path_length(G):.2f}")
    print(f"   网络直径: {nx.diameter(G)}")

    # 集聚系数
    print(f"\n5. 集聚系数:")
    print(f"   平均集聚系数: {nx.average_clustering(G):.4f}")

    # 中心性分析
    betweenness = nx.betweenness_centrality(G)
    top5_betweenness = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
    print(f"\n6. 介数中心性Top5:")
    for node, bc in top5_betweenness:
        print(f"   节点{node}: {bc:.4f}")

    return G

if __name__ == "__main__":
    exercise_full_analysis()
