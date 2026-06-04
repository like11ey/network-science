"""
Phase 8.1 - 机器学习与网络分析的集成
使用scikit-learn预测网络中节点的重要性
结合前7个Phase的知识，展示如何将机器学习应用于复杂网络分析

学习目标：
1. 从网络中提取节点特征（度、集聚系数、中心性等）
2. 构建训练数据集
3. 训练多种分类模型（随机森林、梯度提升、逻辑回归）
4. 评估模型性能（准确率、精确率、召回率、F1）
5. 分析特征重要性
"""

import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 使用非交互式后端，避免GUI依赖
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============================================================
# 1. 节点特征提取
# ============================================================

def extract_node_features(G):
    """
    从网络中提取每个节点的特征向量

    特征包括：
    - degree: 节点度数（邻居数量）
    - clustering: 集聚系数（邻居之间互连的程度）
    - betweenness_centrality: 介数中心性（经过该节点的最短路径比例）
    - closeness_centrality: 接近中心性（到其他节点的平均距离的倒数）
    - pagerank: PageRank值（Google的网页排名算法）
    - kcore: k-core值（节点所在的最深core）
    - degree_centrality: 度中心性（度除以最大可能度）
    - eccentricity: 离心率（到最远节点的距离）

    参数:
        G: networkx.Graph - 输入网络

    返回:
        pandas.DataFrame - 包含所有节点特征的数据框
    """
    print("正在提取节点特征...")

    # 基本度特征
    degrees = dict(G.degree())
    print(f"  [1/8] 度特征提取完成")

    # 集聚系数
    clustering = nx.clustering(G)
    print(f"  [2/8] 集聚系数计算完成")

    # 介数中心性（计算量大，对大网络可采样）
    betweenness = nx.betweenness_centrality(G)
    print(f"  [3/8] 介数中心性计算完成")

    # 接近中心性
    closeness = nx.closeness_centrality(G)
    print(f"  [4/8] 接近中心性计算完成")

    # PageRank
    pagerank = nx.pagerank(G)
    print(f"  [5/8] PageRank计算完成")

    # k-core分解
    core_numbers = nx.core_number(G)
    print(f"  [6/8] k-core分解完成")

    # 度中心性
    degree_centrality = nx.degree_centrality(G)
    print(f"  [7/8] 度中心性计算完成")

    # 离心率（需要连通图，不连通时取最大连通分量）
    if nx.is_connected(G):
        eccentricity = nx.eccentricity(G)
    else:
        # 对最大连通分量计算
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        ecc = nx.eccentricity(subgraph)
        eccentricity = {n: ecc.get(n, 0) for n in G.nodes()}
    print(f"  [8/8] 离心率计算完成")

    # 组装特征矩阵
    features = []
    for node in G.nodes():
        features.append({
            'node': node,
            'degree': degrees[node],
            'clustering': clustering[node],
            'betweenness_centrality': betweenness[node],
            'closeness_centrality': closeness[node],
            'pagerank': pagerank[node],
            'kcore': core_numbers[node],
            'degree_centrality': degree_centrality[node],
            'eccentricity': eccentricity[node],
        })

    df = pd.DataFrame(features)
    print(f"特征提取完成: {len(df)}个节点, {len(df.columns)-1}个特征")
    return df


# ============================================================
# 2. 标签生成：定义"重要节点"
# ============================================================

def label_important_nodes(G, method='degree_top10', threshold=None):
    """
    为节点生成重要性标签

    方法：
    - 'degree_top10': 度最高的10%节点标记为重要
    - 'betweenness_top10': 介数最高的10%节点标记为重要
    - 'threshold': 度超过阈值的节点标记为重要

    参数:
        G: networkx.Graph - 输入网络
        method: str - 标签生成方法
        threshold: int - 度阈值（仅在method='threshold'时使用）

    返回:
        dict - {node: label}，label为1(重要)或0(不重要)
    """
    labels = {}

    if method == 'degree_top10':
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        top_10_pct = max(1, int(len(degrees) * 0.1))
        important_nodes = set(n for n, _ in degrees[:top_10_pct])
        for node in G.nodes():
            labels[node] = 1 if node in important_nodes else 0

    elif method == 'betweenness_top10':
        betweenness = nx.betweenness_centrality(G)
        sorted_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        top_10_pct = max(1, int(len(sorted_bc) * 0.1))
        important_nodes = set(n for n, _ in sorted_bc[:top_10_pct])
        for node in G.nodes():
            labels[node] = 1 if node in important_nodes else 0

    elif method == 'threshold':
        if threshold is None:
            threshold = 10
        for node in G.nodes():
            labels[node] = 1 if G.degree(node) >= threshold else 0

    num_important = sum(labels.values())
    print(f"标签生成完成: {num_important}/{len(labels)} "
          f"({100*num_important/len(labels):.1f}%) 个重要节点")
    return labels


# ============================================================
# 3. 数据集构建
# ============================================================

def build_dataset(G, label_method='degree_top10'):
    """
    构建完整的机器学习数据集

    参数:
        G: networkx.Graph - 输入网络
        label_method: str - 标签生成方法

    返回:
        X: np.array - 特征矩阵
        y: np.array - 标签向量
        feature_names: list - 特征名称列表
    """
    # 提取特征
    df = extract_node_features(G)

    # 生成标签
    labels = label_important_nodes(G, method=label_method)
    df['is_important'] = df['node'].map(labels)

    # 分离特征和标签
    feature_names = ['degree', 'clustering', 'betweenness_centrality',
                     'closeness_centrality', 'pagerank', 'kcore',
                     'degree_centrality', 'eccentricity']

    X = df[feature_names].values
    y = df['is_important'].values

    print(f"\n数据集构建完成:")
    print(f"  特征维度: {X.shape}")
    print(f"  正样本: {sum(y==1)} ({100*sum(y==1)/len(y):.1f}%)")
    print(f"  负样本: {sum(y==0)} ({100*sum(y==0)/len(y):.1f}%)")

    return X, y, feature_names


# ============================================================
# 4. 模型训练与评估
# ============================================================

def train_and_evaluate(X, y, feature_names, test_size=0.3, random_state=42):
    """
    训练多种分类模型并评估性能

    模型：
    - 随机森林（Random Forest）：集成多棵决策树
    - 梯度提升（Gradient Boosting）：逐步修正错误
    - 逻辑回归（Logistic Regression）：线性分类器

    参数:
        X: np.array - 特征矩阵
        y: np.array - 标签向量
        feature_names: list - 特征名称
        test_size: float - 测试集比例
        random_state: int - 随机种子

    返回:
        dict - 包含模型、评估结果和特征重要性
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 特征标准化（逻辑回归需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型
    models = {
        '随机森林': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        '梯度提升': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=random_state
        ),
        '逻辑回归': LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"训练模型: {name}")
        print(f"{'='*50}")

        # 逻辑回归使用标准化特征
        if name == '逻辑回归':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n评估结果:")
        print(f"  准确率 (Accuracy):  {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall):    {recall:.4f}")
        print(f"  F1分数 (F1-Score):  {f1:.4f}")
        print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print(f"\n详细分类报告:")
        print(classification_report(y_test, y_pred,
                                   target_names=['不重要', '重要']))

        # 特征重要性
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            print(f"特征重要性:")
            for feat, imp in sorted(feature_importance.items(),
                                   key=lambda x: -x[1]):
                print(f"  {feat}: {imp:.4f}")
        elif hasattr(model, 'coef_'):
            # 逻辑回归的系数绝对值作为重要性
            importance = np.abs(model.coef_[0])
            feature_importance = dict(zip(feature_names, importance))
            print(f"特征重要性 (|系数|):")
            for feat, imp in sorted(feature_importance.items(),
                                   key=lambda x: -x[1]):
                print(f"  {feat}: {imp:.4f}")

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred,
        }

    return results


# ============================================================
# 5. 可视化结果
# ============================================================

def plot_model_comparison(results, save_path=None):
    """
    绘制模型性能对比图

    参数:
        results: dict - train_and_evaluate的返回结果
        save_path: str - 图片保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：评估指标对比
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        axes[0].bar(x + i * width, values, width, label=name, alpha=0.8)

    axes[0].set_xlabel('评估指标')
    axes[0].set_ylabel('分数')
    axes[0].set_title('模型性能对比')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metric_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：特征重要性（使用随机森林）
    if '随机森林' in results and results['随机森林']['feature_importance']:
        importance = results['随机森林']['feature_importance']
        sorted_feats = sorted(importance.items(), key=lambda x: x[1])
        feats = [f[0] for f in sorted_feats]
        vals = [f[1] for f in sorted_feats]

        axes[1].barh(feats, vals, color='steelblue', alpha=0.8)
        axes[1].set_xlabel('重要性')
        axes[1].set_title('随机森林 - 特征重要性')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n模型对比图已保存: {save_path}")
    plt.close('all')


def plot_confusion_matrices(results, save_path=None):
    """
    绘制混淆矩阵

    参数:
        results: dict - train_and_evaluate的返回结果
        save_path: str - 图片保存路径
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res['y_test'], res['y_pred'])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name}\n混淆矩阵')

        # 标注数值
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                       ha='center', va='center', fontsize=14,
                       color='white' if cm[i, j] > cm.max() / 2 else 'black')

        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['不重要', '重要'])
        ax.set_yticklabels(['不重要', '重要'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵图已保存: {save_path}")
    plt.close('all')


# ============================================================
# 6. 综合实验
# ============================================================

def run_full_experiment(N=500, m=3, label_method='degree_top10',
                        save_dir=None):
    """
    运行完整的机器学习实验

    流程：
    1. 生成BA无标度网络
    2. 提取节点特征
    3. 生成重要性标签
    4. 训练三种模型
    5. 评估并可视化

    参数:
        N: int - 网络节点数
        m: int - BA模型每步连接数
        label_method: str - 标签生成方法
        save_dir: str - 结果保存目录

    返回:
        dict - 实验结果
    """
    print("=" * 60)
    print("复杂网络节点重要性预测 - 机器学习实验")
    print("=" * 60)

    # Step 1: 生成网络
    print(f"\n--- Step 1: 生成BA无标度网络 (N={N}, m={m}) ---")
    G = nx.barabasi_albert_graph(N, m, seed=42)
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  是否连通: {nx.is_connected(G)}")

    # Step 2 & 3: 构建数据集
    print(f"\n--- Step 2: 构建数据集 ---")
    X, y, feature_names = build_dataset(G, label_method=label_method)

    # Step 4: 训练和评估模型
    print(f"\n--- Step 3: 训练和评估模型 ---")
    results = train_and_evaluate(X, y, feature_names)

    # Step 5: 可视化
    print(f"\n--- Step 4: 生成可视化 ---")
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plot_model_comparison(results,
                            save_path=f"{save_dir}/ml_model_comparison.png")
        plot_confusion_matrices(results,
                              save_path=f"{save_dir}/ml_confusion_matrices.png")
    else:
        plot_model_comparison(results)
        plot_confusion_matrices(results)

    # 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"\n网络: BA(N={N}, m={m})")
    print(f"标签方法: {label_method}")
    print(f"\n模型性能排名 (按F1分数):")
    ranked = sorted(results.items(), key=lambda x: -x[1]['f1'])
    for i, (name, res) in enumerate(ranked, 1):
        print(f"  {i}. {name}: F1={res['f1']:.4f}, "
              f"Accuracy={res['accuracy']:.4f}")

    # 找出最重要的特征
    best_model_name = ranked[0][0]
    best_result = ranked[0][1]
    if best_result['feature_importance']:
        top_feat = max(best_result['feature_importance'].items(),
                      key=lambda x: x[1])
        print(f"\n最重要特征: {top_feat[0]} (重要性={top_feat[1]:.4f})")

    return results


# ============================================================
# 练习
# ============================================================

def exercise_custom_features():
    """
    练习：自定义节点特征并重新训练模型

    任务：
    1. 添加新的特征（如三角形数量、度的对数等）
    2. 使用不同的标签方法（如betweenness_top10）
    3. 对比不同网络规模下的模型表现
    """
    print("\n" + "=" * 60)
    print("练习：自定义特征工程")
    print("=" * 60)

    # 生成网络
    G = nx.barabasi_albert_graph(300, 3, seed=42)

    # 提取基础特征
    df = extract_node_features(G)

    # ---- 自定义特征 ----
    # 特征1: 节点参与的三角形数量
    triangles = nx.triangles(G)
    df['triangles'] = df['node'].map(triangles)

    # 特征2: 度的对数变换（处理度分布的长尾特性）
    df['log_degree'] = np.log1p(df['degree'])

    # 特征3: 邻居的平均度（度同配性）
    avg_neighbor_degree = nx.average_neighbor_degree(G)
    df['avg_neighbor_degree'] = df['node'].map(avg_neighbor_degree)

    # 使用betweenness_top10作为标签
    labels = label_important_nodes(G, method='betweenness_top10')
    df['is_important'] = df['node'].map(labels)

    # 扩展后的特征列表
    extended_features = ['degree', 'clustering', 'betweenness_centrality',
                        'closeness_centrality', 'pagerank', 'kcore',
                        'degree_centrality', 'eccentricity',
                        'triangles', 'log_degree', 'avg_neighbor_degree']

    X = df[extended_features].values
    y = df['is_important'].values

    print(f"\n扩展特征集: {len(extended_features)} 个特征")
    print(f"新增特征: triangles, log_degree, avg_neighbor_degree")

    # 训练随机森林
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"\n扩展特征后的随机森林性能:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")

    # 特征重要性
    importance = dict(zip(extended_features, rf.feature_importances_))
    print(f"\n特征重要性排名:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s}: {imp:.4f} {bar}")


def exercise_different_networks():
    """
    练习：在不同类型的网络上测试模型泛化能力

    任务：
    1. 在ER随机网络上训练
    2. 在WS小世界网络上测试
    3. 分析模型的泛化能力
    """
    print("\n" + "=" * 60)
    print("练习：跨网络类型泛化测试")
    print("=" * 60)

    # 生成不同类型的网络
    ba = nx.barabasi_albert_graph(500, 3, seed=42)
    er = nx.erdos_renyi_graph(500, 0.012, seed=42)
    ws = nx.watts_strogatz_graph(500, 6, 0.3, seed=42)

    networks = {
        'BA无标度': ba,
        'ER随机': er,
        'WS小世界': ws,
    }

    # 在每种网络上训练和评估
    feature_names = ['degree', 'clustering', 'betweenness_centrality',
                     'closeness_centrality', 'pagerank', 'kcore',
                     'degree_centrality', 'eccentricity']

    print(f"\n{'网络类型':<12} {'节点数':<8} {'边数':<8} {'F1分数':<10}")
    print("-" * 40)

    for name, G in networks.items():
        X, y, _ = build_dataset(G, label_method='degree_top10')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        print(f"{name:<12} {G.number_of_nodes():<8} "
              f"{G.number_of_edges():<8} {f1:.4f}")

    print("\n结论: 随机森林在不同网络类型上都能有效预测重要节点")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 运行主实验
    results = run_full_experiment(
        N=500, m=3,
        label_method='degree_top10',
        save_dir='E:/ai_python/results'
    )

    # 运行练习
    exercise_custom_features()
    exercise_different_networks()

    print("\n" + "=" * 60)
    print("Phase 8 完成！")
    print("=" * 60)
    print("\n学习要点总结:")
    print("1. 网络节点可以提取丰富的结构特征")
    print("2. 集成学习方法（随机森林、梯度提升）通常表现最好")
    print("3. 特征工程对模型性能有重要影响")
    print("4. 在复杂网络中，度和中心性是最有区分力的特征")
    print("5. 这些技术可以应用于实际的网络关键节点识别问题")
