"""
Phase 1.5 - 文件读写
CSV、JSON、网络数据的读写
"""

import csv
import json
import os
from pathlib import Path

# ============================================================
# 1. 基本文件操作
# ============================================================

# 写文件（Python会自动处理编码）
data_dir = Path("E:/ai_python/data")
data_dir.mkdir(exist_ok=True)

# 写入文本
with open(data_dir / "sample_network.txt", "w", encoding="utf-8") as f:
    f.write("# 简单网络数据\n")
    f.write("# 节点数: 5, 边数: 6\n")
    f.write("0 1\n")
    f.write("0 2\n")
    f.write("1 2\n")
    f.write("2 3\n")
    f.write("3 4\n")

# 读取文本
with open(data_dir / "sample_network.txt", "r", encoding="utf-8") as f:
    content = f.read()
print("读取文件:")
print(content)

# 逐行读取（大文件推荐）
print("逐行读取:")
with open(data_dir / "sample_network.txt", "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if line and not line.startswith("#"):
            print(f"  第{line_num}行: {line}")

# ============================================================
# 2. CSV文件操作
# ============================================================

# 写入CSV
edges_data = [
    {"source": 0, "target": 1, "weight": 1.0},
    {"source": 0, "target": 2, "weight": 0.8},
    {"source": 1, "target": 2, "weight": 1.2},
    {"source": 2, "target": 3, "weight": 0.5},
    {"source": 3, "target": 4, "weight": 1.0},
]

csv_path = data_dir / "edges.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "target", "weight"])
    writer.writeheader()
    writer.writerows(edges_data)
print(f"\nCSV已写入: {csv_path}")

# 读取CSV
print("\n读取CSV:")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['source']} -> {row['target']}, 权重={row['weight']}")

# ============================================================
# 3. JSON文件操作（网络数据常用格式）
# ============================================================

# 网络数据结构
network_data = {
    "name": "示例网络",
    "nodes": [
        {"id": 0, "label": "节点A"},
        {"id": 1, "label": "节点B"},
        {"id": 2, "label": "节点C"},
        {"id": 3, "label": "节点D"},
        {"id": 4, "label": "节点E"},
    ],
    "edges": [
        {"source": 0, "target": 1},
        {"source": 0, "target": 2},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
    ],
    "properties": {
        "num_nodes": 5,
        "num_edges": 5,
        "directed": False
    }
}

# 写入JSON
json_path = data_dir / "network.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(network_data, f, ensure_ascii=False, indent=2)
print(f"\nJSON已写入: {json_path}")

# 读取JSON
with open(json_path, "r", encoding="utf-8") as f:
    loaded_data = json.load(f)
print(f"JSON读取: {loaded_data['name']}")
print(f"节点数: {loaded_data['properties']['num_nodes']}")

# ============================================================
# 4. Python内置的序列化（pickle）
# ============================================================

import pickle

# 保存Python对象
data = {
    "degree_distribution": {1: 10, 2: 25, 3: 30, 4: 20, 5: 15},
    "clustering_coefficient": 0.35,
    "avg_path_length": 3.2
}

pickle_path = data_dir / "analysis_result.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(data, f)
print(f"\nPickle已写入: {pickle_path}")

# 读取Python对象
with open(pickle_path, "rb") as f:
    loaded = pickle.load(f)
print(f"Pickle读取: {loaded}")

# ============================================================
# 5. pathlib路径操作（比os.path更现代）
# ============================================================

print(f"\n路径操作:")
print(f"  数据目录: {data_dir}")
print(f"  目录存在: {data_dir.exists()}")
print(f"  列出文件: {[f.name for f in data_dir.iterdir()]}")

# ============================================================
# 练习：读取边列表并构建邻接表
# ============================================================

def exercise_load_network():
    """从边列表文件加载网络并构建邻接表"""
    from collections import defaultdict

    # 读取边列表
    edges = []
    with open(data_dir / "edges.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((int(row["source"]), int(row["target"])))

    # 构建邻接表
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # 分析网络
    print("\n网络分析:")
    print(f"  节点数: {len(adj)}")
    print(f"  边数: {len(edges)}")
    print(f"  平均度: {sum(len(n) for n in adj.values()) / len(adj):.2f}")

    # 保存分析结果
    result = {
        "num_nodes": len(adj),
        "num_edges": len(edges),
        "adjacency": dict(adj)
    }
    with open(data_dir / "network_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  结果已保存: {data_dir / 'network_analysis.json'}")

if __name__ == "__main__":
    exercise_load_network()
