"""
tests/test_ml_prediction.py - Phase 8 机器学习预测 针对性测试

覆盖核心函数：
- extract_node_features: 节点特征提取
- label_important_nodes: 重要性标签生成
- build_dataset: 数据集构建
- train_and_evaluate: 模型训练与评估
"""

import sys
import os
import importlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import networkx as nx
import numpy as np
import pandas as pd


def _import_from(subpackage, module_filename, name):
    module = importlib.import_module(f"{subpackage}.{module_filename}")
    return getattr(module, name)


def _get_extract_node_features():
    return _import_from("phase8_ml_integration", "01_ml_prediction", "extract_node_features")


def _get_label_important_nodes():
    return _import_from("phase8_ml_integration", "01_ml_prediction", "label_important_nodes")


def _get_build_dataset():
    return _import_from("phase8_ml_integration", "01_ml_prediction", "build_dataset")


def _get_train_and_evaluate():
    return _import_from("phase8_ml_integration", "01_ml_prediction", "train_and_evaluate")


# ============================================================
# extract_node_features 特征提取
# ============================================================

class TestExtractNodeFeatures:
    """测试节点特征提取的行为不变量"""

    def test_output_row_count_equals_nodes(self):
        """返回的 DataFrame 行数 = 节点数"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        df = extract(G)
        assert len(df) == 50

    def test_output_contains_all_feature_columns(self):
        """返回的 DataFrame 应包含所有8个特征列"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        df = extract(G)
        expected_cols = {'node', 'degree', 'clustering', 'betweenness_centrality',
                         'closeness_centrality', 'pagerank', 'kcore',
                         'degree_centrality', 'eccentricity'}
        assert expected_cols.issubset(set(df.columns))

    def test_degree_values_correct(self):
        """degree 列应与 networkx 计算的度一致"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        df = extract(G)
        for _, row in df.iterrows():
            assert row['degree'] == G.degree(int(row['node']))

    def test_feature_values_non_negative(self):
        """所有特征值应 >= 0"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        df = extract(G)
        feature_cols = ['degree', 'clustering', 'betweenness_centrality',
                        'closeness_centrality', 'pagerank', 'kcore',
                        'degree_centrality', 'eccentricity']
        for col in feature_cols:
            assert (df[col] >= 0).all(), f"列 {col} 存在负值"

    def test_pagerank_sums_to_one(self):
        """PageRank 值之和应约等于 1"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        df = extract(G)
        assert abs(df['pagerank'].sum() - 1.0) < 1e-6

    def test_clustering_coefficient_bounded(self):
        """集聚系数应在 [0, 1] 范围内"""
        extract = _get_extract_node_features()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        df = extract(G)
        assert (df['clustering'] >= 0).all()
        assert (df['clustering'] <= 1).all()

    def test_disconnected_graph_handled(self):
        """非连通图也应能正常提取特征"""
        extract = _get_extract_node_features()
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_edges_from([(3, 4), (4, 5)])
        df = extract(G)
        assert len(df) == 6
        assert 'eccentricity' in df.columns


# ============================================================
# label_important_nodes 标签生成
# ============================================================

class TestLabelImportantNodes:
    """测试重要性标签生成的行为不变量"""

    def test_degree_top10_label_count(self):
        """degree_top10 方法应标记约 10% 的节点为重要"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        labels = label_fn(G, method='degree_top10')
        num_important = sum(labels.values())
        # 10% of 100 = 10
        assert num_important == 10

    def test_labels_are_binary(self):
        """所有标签应为 0 或 1"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        labels = label_fn(G, method='degree_top10')
        for node, label in labels.items():
            assert label in (0, 1)

    def test_labels_cover_all_nodes(self):
        """标签应覆盖所有节点"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        labels = label_fn(G, method='degree_top10')
        assert set(labels.keys()) == set(G.nodes())

    def test_threshold_method(self):
        """threshold 方法：度 >= 阈值的标记为重要"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        threshold = 5
        labels = label_fn(G, method='threshold', threshold=threshold)
        for node in G.nodes():
            expected = 1 if G.degree(node) >= threshold else 0
            assert labels[node] == expected

    def test_betweenness_top10_produces_labels(self):
        """betweenness_top10 方法应产生有效标签"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        labels = label_fn(G, method='betweenness_top10')
        num_important = sum(labels.values())
        assert num_important >= 1
        assert num_important <= len(G) // 2

    def test_high_degree_nodes_labeled_important(self):
        """度最大的节点应被标记为重要"""
        label_fn = _get_label_important_nodes()
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        labels = label_fn(G, method='degree_top10')
        # 度最大的节点应在重要集合中
        hub = max(G.degree(), key=lambda x: x[1])[0]
        assert labels[hub] == 1


# ============================================================
# build_dataset 数据集构建
# ============================================================

class TestBuildDataset:
    """测试数据集构建的行为不变量"""

    def test_X_shape(self):
        """特征矩阵 X 的形状 = (N, 8)"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        X, y, feature_names = build(G)
        assert X.shape == (50, 8)

    def test_y_length_matches_X(self):
        """标签向量 y 的长度应等于 X 的行数"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        X, y, _ = build(G)
        assert len(y) == X.shape[0]

    def test_y_contains_both_classes(self):
        """标签应同时包含 0 和 1"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        X, y, _ = build(G)
        assert 0 in y
        assert 1 in y

    def test_feature_names_count(self):
        """应返回 8 个特征名"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        _, _, feature_names = build(G)
        assert len(feature_names) == 8

    def test_X_values_finite(self):
        """特征矩阵不应包含 NaN 或 Inf"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        X, _, _ = build(G)
        assert np.all(np.isfinite(X))


# ============================================================
# train_and_evaluate 模型训练
# ============================================================

class TestTrainAndEvaluate:
    """测试模型训练与评估的行为不变量"""

    def _get_small_dataset(self):
        """获取一个小型测试数据集"""
        build = _get_build_dataset()
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        X, y, feature_names = build(G)
        return X, y, feature_names

    def test_returns_dict_with_three_models(self):
        """应返回包含三种模型的字典"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        assert len(results) == 3
        assert '随机森林' in results
        assert '梯度提升' in results
        assert '逻辑回归' in results

    def test_each_model_has_required_keys(self):
        """每个模型结果应包含所有必要的评估指标"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        required_keys = {'model', 'accuracy', 'precision', 'recall', 'f1',
                         'cv_mean', 'cv_std', 'feature_importance', 'y_test', 'y_pred'}
        for model_name, model_result in results.items():
            assert required_keys.issubset(set(model_result.keys())), \
                f"模型 {model_name} 缺少键: {required_keys - set(model_result.keys())}"

    def test_metrics_in_valid_range(self):
        """所有评估指标应在 [0, 1] 范围内"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        for model_name, res in results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                assert 0 <= res[metric] <= 1, \
                    f"模型 {model_name} 的 {metric}={res[metric]} 超出 [0,1]"

    def test_accuracy_above_chance(self):
        """模型准确率应显著高于随机猜测（>0.5）"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        for model_name, res in results.items():
            assert res['accuracy'] > 0.5, \
                f"模型 {model_name} 准确率 {res['accuracy']:.4f} 低于随机水平"

    def test_feature_importance_present_for_tree_models(self):
        """树模型应有特征重要性"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        # 随机森林和梯度提升应有 feature_importances_
        for model_name in ['随机森林', '梯度提升']:
            assert results[model_name]['feature_importance'] is not None
            assert len(results[model_name]['feature_importance']) == len(names)

    def test_cv_scores_reasonable(self):
        """交叉验证均值应在 [0, 1] 范围内"""
        train = _get_train_and_evaluate()
        X, y, names = self._get_small_dataset()
        results = train(X, y, names)
        for model_name, res in results.items():
            assert 0 <= res['cv_mean'] <= 1
            assert res['cv_std'] >= 0
