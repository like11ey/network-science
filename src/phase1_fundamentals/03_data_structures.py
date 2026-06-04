"""
Phase 1.3 - Python数据结构
列表、字典、集合、元组 - 比C/Java的容器更强大
"""

# ============================================================
# 1. 列表（List）- 类似Java的ArrayList，但更灵活
# ============================================================

# 创建列表
nodes = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # 可以混合类型（C/Java不行）

# 切片操作 - Python最强大的特性之一
print("切片操作:")
print(f"  前3个: {nodes[:3]}")      # [1, 2, 3]
print(f"  后2个: {nodes[-2:]}")     # [4, 5]
print(f"  中间: {nodes[1:4]}")      # [2, 3, 4]
print(f"  步长: {nodes[::2]}")      # [1, 3, 5]
print(f"  反转: {nodes[::-1]}")     # [5, 4, 3, 2, 1]

# 列表方法
edges = [10, 20, 30]
edges.append(40)          # 末尾添加
edges.insert(0, 5)       # 指定位置插入
edges.extend([50, 60])   # 扩展列表
print(f"\n列表操作: {edges}")

edges.remove(30)          # 删除第一个匹配值
popped = edges.pop()     # 弹出最后一个
print(f"删除后: {edges}, 弹出: {popped}")

# 列表排序
degrees = [3, 1, 4, 1, 5, 9, 2, 6]
degrees.sort()            # 原地排序
print(f"排序: {degrees}")
degrees.sort(reverse=True)
print(f"降序: {degrees}")

# ============================================================
# 2. 字典（Dict）- 类似Java的HashMap，但语法更简洁
# ============================================================

# 创建字典
network_stats = {
    "nodes": 1000,
    "edges": 5000,
    "avg_degree": 10.0,
    "clustering": 0.35
}

# 访问和修改
print(f"\n字典访问: 节点数={network_stats['nodes']}")
network_stats["density"] = 0.01  # 添加新键值对
del network_stats["density"]     # 删除键值对

# 安全访问（避免KeyError）
avg_path = network_stats.get("avg_path_length", "未计算")
print(f"安全访问: {avg_path}")

# 字典遍历
print("\n字典遍历:")
for key, value in network_stats.items():
    print(f"  {key}: {value}")

# 字典推导式
squares_dict = {x: x**2 for x in range(6)}
print(f"\n字典推导式: {squares_dict}")

# 实际应用：统计度分布
from collections import Counter
degrees = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
degree_dist = dict(Counter(degrees))
print(f"度分布: {degree_dist}")

# ============================================================
# 3. 集合（Set）- 类似Java的HashSet
# ============================================================

# 创建集合
neighbors_1 = {1, 2, 3, 4, 5}
neighbors_2 = {3, 4, 5, 6, 7}

# 集合运算 - 这些在社交网络分析中非常有用
print(f"\n集合运算:")
print(f"  交集(共同邻居): {neighbors_1 & neighbors_2}")
print(f"  并集(所有邻居): {neighbors_1 | neighbors_2}")
print(f"  差集(独有邻居): {neighbors_1 - neighbors_2}")
print(f"  对称差: {neighbors_1 ^ neighbors_2}")

# 集合推导式
even_set = {x for x in range(20) if x % 2 == 0}
print(f"偶数集合: {even_set}")

# ============================================================
# 4. 元组（Tuple）- 不可变的列表
# ============================================================

# 元组用圆括号
node = (1, 2)  # 节点坐标
edge = (0, 1)  # 边

# 元组解包
x, y = node
print(f"\n元组解包: x={x}, y={y}")

# 函数返回多个值（实际返回的是元组）
def get_network_info():
    return 1000, 5000, 10.0  # 返回三个值

nodes, edges, avg_degree = get_network_info()
print(f"网络信息: {nodes}节点, {edges}边, 平均度{avg_degree}")

# ============================================================
# 5. 默认字典（collections.defaultdict）
# ============================================================

from collections import defaultdict

# 按度值分组节点
adj_list = {
    0: [1, 2, 3],
    1: [0, 2],
    2: [0, 1, 3, 4],
    3: [0, 2],
    4: [2]
}

# 按度值分组
degree_groups = defaultdict(list)
for node, neighbors in adj_list.items():
    degree_groups[len(neighbors)].append(node)

print(f"\n按度值分组:")
for deg, nodes in sorted(degree_groups.items()):
    print(f"  度={deg}: 节点{nodes}")

# ============================================================
# 6. 常用内置函数
# ============================================================

numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(f"\n内置函数:")
print(f"  长度: {len(numbers)}")
print(f"  最大值: {max(numbers)}")
print(f"  最小值: {min(numbers)}")
print(f"  求和: {sum(numbers)}")
print(f"  排序: {sorted(numbers)}")
print(f"  去重排序: {sorted(set(numbers))}")
print(f"  同时获取最小最大: {min(numbers), max(numbers)}")

# map和filter（更Pythonic的方式是列表推导式）
# 传统方式
mapped = list(map(lambda x: x**2, numbers))
# Pythonic方式
mapped_pythonic = [x**2 for x in numbers]
print(f"\nmap结果: {mapped}")
print(f"推导式结果: {mapped_pythonic}")

# ============================================================
# 练习：构建一个简单的邻接表
# ============================================================

def exercise_adjacency_list():
    """
    从边列表构建邻接表
    边列表: [(0,1), (0,2), (1,2), (2,3), (3,4)]
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]

    # 用defaultdict构建无向图邻接表
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    print("\n邻接表:")
    for node in sorted(adj.keys()):
        print(f"  {node}: {sorted(adj[node])}")

    # 计算每个节点的度
    degrees = {node: len(neighbors) for node, neighbors in adj.items()}
    print(f"度序列: {degrees}")
    print(f"平均度: {sum(degrees.values()) / len(degrees):.2f}")

if __name__ == "__main__":
    exercise_adjacency_list()
