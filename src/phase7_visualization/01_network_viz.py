"""
Phase 7.1 - 网络可视化
使用matplotlib和networkx绘制专业网络图
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目根目录（自动定位）
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 基础网络可视化
# ============================================================

def basic_network_viz(G, title="网络结构", save_path=None):
    """基础网络可视化"""
    plt.figure(figsize=(8, 6))

    # layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # 节点大小与度成正比
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 30 + 50 for n in G.nodes()]

    # 节点颜色与度成正比
    node_colors = [degrees[n] for n in G.nodes()]

    # 绘制
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap=plt.cm.YlOrRd,
                                   alpha=0.8)
    plt.colorbar(nodes, label='度')
    plt.title(title, fontsize=14)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")
    plt.close()

# ============================================================
# 2. 度分布图
# ============================================================

def plot_degree_distribution(G, model_name, ax=None, save_path=None):
    """绘制度分布（对数坐标）"""
    degrees = [d for _, d in G.degree()]
    from collections import Counter
    degree_dist = Counter(degrees)

    k_vals = sorted(degree_dist.keys())
    p_vals = [degree_dist[k] / len(degrees) for k in k_vals]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(k_vals, p_vals, s=30, alpha=0.7, color='steelblue')
    ax.set_xlabel('度 k', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title(f'{model_name} - 度分布', fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

# ============================================================
# 3. 级联失效过程可视化
# ============================================================

def visualize_cascade_process(G, failed_nodes_history, title="级联过程",
                              save_path=None):
    """
    可视化级联失效过程
    failed_nodes_history: 每一步失效的节点列表
    """
    steps = len(failed_nodes_history)
    cols = min(steps, 4)
    rows = (steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for step in range(steps):
        ax = axes[step]
        pos = nx.spring_layout(G, seed=42, k=0.3)

        # 当前存活的节点
        failed_set = set()
        for i in range(step + 1):
            failed_set.update(failed_nodes_history[i])

        alive_nodes = [n for n in G.nodes() if n not in failed_set]
        failed_in_step = failed_nodes_history[step]

        # 绘制边（只绘制存活节点之间的边）
        alive_edges = [(u, v) for u, v in G.edges() if u in alive_nodes and v in alive_nodes]
        nx.draw_networkx_edges(G, pos, edgelist=alive_edges, ax=ax,
                               alpha=0.2, width=0.3, edge_color='gray')

        # 绘制存活节点
        if alive_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=alive_nodes, ax=ax,
                                   node_size=30, node_color='steelblue', alpha=0.7)

        # 绘制本步失效的节点（红色）
        if failed_in_step:
            nx.draw_networkx_nodes(G, pos, nodelist=failed_in_step, ax=ax,
                                   node_size=50, node_color='red', alpha=0.9)

        ax.set_title(f'步骤 {step+1}: 失效{len(failed_in_step)}个节点', fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(steps, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"级联过程图已保存: {save_path}")
    plt.close()

# ============================================================
# 4. 综合分析仪表板
# ============================================================

def create_analysis_dashboard(G, model_name, save_path=None):
    """创建综合分析仪表板"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 网络结构
    ax1 = fig.add_subplot(gs[0, 0])
    pos = nx.spring_layout(G, seed=42, k=0.5)
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 20 + 30 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_sizes,
                           node_color=[degrees[n] for n in G.nodes()],
                           cmap=plt.cm.YlOrRd, alpha=0.8)
    ax1.set_title('网络结构', fontsize=12)
    ax1.axis('off')

    # 2. 度分布
    ax2 = fig.add_subplot(gs[0, 1])
    from collections import Counter
    degree_dist = Counter([d for _, d in G.degree()])
    k_vals = sorted(degree_dist.keys())
    p_vals = [degree_dist[k] / G.number_of_nodes() for k in k_vals]
    ax2.scatter(k_vals, p_vals, s=20, alpha=0.7, color='steelblue')
    ax2.set_xlabel('度 k')
    ax2.set_ylabel('P(k)')
    ax2.set_title('度分布', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. 度直方图
    ax3 = fig.add_subplot(gs[0, 2])
    degrees_list = [d for _, d in G.degree()]
    ax3.hist(degrees_list, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xlabel('度')
    ax3.set_ylabel('频次')
    ax3.set_title('度直方图', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. 集聚系数分布
    ax4 = fig.add_subplot(gs[1, 0])
    clustering = [nx.clustering(G, n) for n in G.nodes()]
    ax4.hist(clustering, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax4.set_xlabel('集聚系数')
    ax4.set_ylabel('频次')
    ax4.set_title('集聚系数分布', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. 介数中心性分布
    ax5 = fig.add_subplot(gs[1, 1])
    betweenness = list(nx.betweenness_centrality(G).values())
    ax5.hist(betweenness, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax5.set_xlabel('介数中心性')
    ax5.set_ylabel('频次')
    ax5.set_title('介数中心性分布', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 6. 统计信息
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    info_text = f"""网络统计信息
{'='*30}
模型: {model_name}
节点数: {G.number_of_nodes()}
边数: {G.number_of_edges()}
密度: {nx.density(G):.4f}
平均度: {np.mean(degrees_list):.2f}
最大度: {max(degrees_list)}
平均集聚系数: {np.mean(clustering):.4f}
平均介数: {np.mean(betweenness):.4f}"""

    if nx.is_connected(G):
        info_text += f"\n平均路径长度: {nx.average_shortest_path_length(G):.2f}"
        info_text += f"\n直径: {nx.diameter(G)}"

    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{model_name}网络综合分析', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"仪表板已保存: {save_path}")
    plt.close()

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 生成网络
    ba = nx.barabasi_albert_graph(500, 3, seed=42)
    er = nx.erdos_renyi_graph(500, 0.02, seed=42)

    # 基础可视化
    basic_network_viz(ba, "BA无标度网络", str(RESULTS_DIR / "ba_network.png"))
    basic_network_viz(er, "ER随机网络", str(RESULTS_DIR / "er_network.png"))

    # 综合仪表板
    create_analysis_dashboard(ba, "BA无标度", str(RESULTS_DIR / "ba_dashboard.png"))
    create_analysis_dashboard(er, "ER随机", str(RESULTS_DIR / "er_dashboard.png"))

    print("\n所有可视化图表已保存到 results/ 目录")
