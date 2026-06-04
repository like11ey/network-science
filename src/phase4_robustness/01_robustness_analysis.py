"""
Phase 4.1 - 网络鲁棒性分析
模拟随机故障和蓄意攻击，验证渗流理论
对应段老师研究：Albert et al. (2000) Nature
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 节点移除策略
# ============================================================

def random_removal(G, fraction):
    """
    随机故障：随机移除指定比例的节点
    对应：随机故障（random failure）
    """
    nodes = list(G.nodes())
    num_remove = int(len(nodes) * fraction)
    nodes_to_remove = random.sample(nodes, num_remove)
    G_copy = G.copy()
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

def targeted_removal_by_degree(G, fraction):
    """
    蓄意攻击：按度从大到小移除节点
    对应：蓄意攻击（intentional attack）
    """
    # 按度排序
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    num_remove = int(len(degree_sorted) * fraction)
    nodes_to_remove = [node for node, _ in degree_sorted[:num_remove]]
    G_copy = G.copy()
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

def targeted_removal_by_betweenness(G, fraction):
    """
    蓄意攻击：按介数从大到小移除节点
    """
    betweenness = nx.betweenness_centrality(G)
    betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    num_remove = int(len(betweenness_sorted) * fraction)
    nodes_to_remove = [node for node, _ in betweenness_sorted[:num_remove]]
    G_copy = G.copy()
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

# ============================================================
# 2. 渗流模拟
# ============================================================

def percolation_simulation(G, removal_func, fractions, num_trials=10):
    """
    渗流模拟
    返回每个移除比例下的巨连通分量比例
    """
    results = []
    n = G.number_of_nodes()

    for f in fractions:
        giant_component_sizes = []

        for trial in range(num_trials):
            # 移除节点
            G_removed = removal_func(G, f)

            # 计算巨连通分量
            if G_removed.number_of_nodes() == 0:
                giant_component_sizes.append(0)
                continue

            largest_cc = max(nx.connected_components(G_removed), key=len)
            giant_size = len(largest_cc) / n
            giant_component_sizes.append(giant_size)

        avg_giant_size = np.mean(giant_component_sizes)
        results.append(avg_giant_size)

    return results

# ============================================================
# 3. 相变分析
# ============================================================

def find_critical_threshold(fractions, giant_sizes):
    """
    通过数值方法寻找临界阈值 f_c
    巨连通分量大小从 >0.01 降到 <0.01 的点
    """
    for i, (f, s) in enumerate(zip(fractions, giant_sizes)):
        if s < 0.01:
            if i > 0:
                # 线性插值
                f1, s1 = fractions[i-1], giant_sizes[i-1]
                f2, s2 = f, s
                f_c = f1 + (0.01 - s1) * (f2 - f1) / (s2 - s1)
                return f_c
            return f
    return 1.0  # 网络始终连通

# ============================================================
# 4. Malloy-Reed判据验证
# ============================================================

def verify_malloy_reed(G):
    """
    验证Malloy-Reed判据
    κ = <k²> / <k> > 2 时，巨连通分量存在
    """
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)
    k_sq_mean = np.mean([d**2 for d in degrees])
    kappa = k_sq_mean / k_mean

    print(f"\nMalloy-Reed判据验证:")
    print(f"  <k> = {k_mean:.2f}")
    print(f"  <k^2> = {k_sq_mean:.2f}")
    print(f"  kappa = <k^2>/<k> = {kappa:.2f}")
    print(f"  判据条件 κ > 2: {'满足' if kappa > 2 else '不满足'}")
    print(f"  巨连通分量{'存在' if kappa > 2 else '不存在'}")

    return kappa

# ============================================================
# 5. 鲁棒性曲线绘制
# ============================================================

def plot_robustness_curves(fractions, results_random, results_degree, results_betweenness,
                          model_name, save_path=None):
    """绘制鲁棒性对比曲线"""
    plt.figure(figsize=(10, 6))

    plt.plot(fractions, results_random, 'o-', label='随机故障', color='blue', linewidth=2)
    plt.plot(fractions, results_degree, 's-', label='蓄意攻击(度)', color='red', linewidth=2)
    plt.plot(fractions, results_betweenness, '^-', label='蓄意攻击(介数)', color='green', linewidth=2)

    plt.xlabel('移除节点比例 f', fontsize=12)
    plt.ylabel('巨连通分量比例 P∞', fontsize=12)
    plt.title(f'{model_name}网络鲁棒性', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"鲁棒性曲线已保存: {save_path}")
    plt.close()

# ============================================================
# 6. 主程序：对比ER和BA网络的鲁棒性
# ============================================================

def main():
    random.seed(42)
    np.random.seed(42)

    N = 500
    fractions = np.linspace(0, 0.5, 50)

    # --- ER随机网络 ---
    print("="*50)
    print("ER随机网络鲁棒性分析")
    print("="*50)
    er = nx.erdos_renyi_graph(N, 0.02, seed=42)
    print(f"节点数: {er.number_of_nodes()}, 边数: {er.number_of_edges()}")
    print(f"平均度: {2*er.number_of_edges()/er.number_of_nodes():.2f}")

    verify_malloy_reed(er)

    er_random = percolation_simulation(er, random_removal, fractions)
    er_degree = percolation_simulation(er, targeted_removal_by_degree, fractions)
    er_betweenness = percolation_simulation(er, targeted_removal_by_betweenness, fractions)

    fc_er_random = find_critical_threshold(fractions, er_random)
    fc_er_degree = find_critical_threshold(fractions, er_degree)
    print(f"\nER网络临界阈值:")
    print(f"  随机故障 f_c ≈ {fc_er_random:.3f}")
    print(f"  蓄意攻击(度) f_c ≈ {fc_er_degree:.3f}")

    plot_robustness_curves(fractions, er_random, er_degree, er_betweenness,
                          "ER随机", "E:/ai_python/results/er_robustness.png")

    # --- BA无标度网络 ---
    print("\n" + "="*50)
    print("BA无标度网络鲁棒性分析")
    print("="*50)
    ba = nx.barabasi_albert_graph(N, 3, seed=42)
    print(f"节点数: {ba.number_of_nodes()}, 边数: {ba.number_of_edges()}")
    print(f"平均度: {2*ba.number_of_edges()/ba.number_of_nodes():.2f}")

    verify_malloy_reed(ba)

    ba_random = percolation_simulation(ba, random_removal, fractions)
    ba_degree = percolation_simulation(ba, targeted_removal_by_degree, fractions)
    ba_betweenness = percolation_simulation(ba, targeted_removal_by_betweenness, fractions)

    fc_ba_random = find_critical_threshold(fractions, ba_random)
    fc_ba_degree = find_critical_threshold(fractions, ba_degree)
    print(f"\nBA网络临界阈值:")
    print(f"  随机故障 f_c ≈ {fc_ba_random:.3f}")
    print(f"  蓄意攻击(度) f_c ≈ {fc_ba_degree:.3f}")

    plot_robustness_curves(fractions, ba_random, ba_degree, ba_betweenness,
                          "BA无标度", "E:/ai_python/results/ba_robustness.png")

    # --- 对比图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(fractions, er_random, 'b--', label='ER随机(随机故障)', linewidth=2)
    plt.plot(fractions, er_degree, 'r--', label='ER随机(蓄意攻击)', linewidth=2)
    plt.plot(fractions, ba_random, 'b-', label='BA无标度(随机故障)', linewidth=2)
    plt.plot(fractions, ba_degree, 'r-', label='BA无标度(蓄意攻击)', linewidth=2)
    plt.xlabel('移除节点比例 f', fontsize=12)
    plt.ylabel('巨连通分量比例 P∞', fontsize=12)
    plt.title('ER vs BA 网络鲁棒性对比', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.5)
    plt.ylim(0, 1.05)
    plt.savefig("E:/ai_python/results/er_vs_ba_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 总结 ---
    print("\n" + "="*50)
    print("实验结论")
    print("="*50)
    print("1. BA无标度网络对随机故障高度鲁棒")
    print(f"   随机故障临界阈值 f_c ≈ {fc_ba_random:.3f}（非常高）")
    print("2. BA无标度网络对蓄意攻击极其脆弱")
    print(f"   蓄意攻击临界阈值 f_c ≈ {fc_ba_degree:.3f}（非常低）")
    print("3. 这就是无标度网络的'阿喀琉斯之踵'")
    print("4. ER随机网络在两种攻击下表现相似（齐次性）")

if __name__ == "__main__":
    main()
