"""
Phase 1.7 - Python高级特性
这些是Python面试常考点，也是写工程代码的必备技能
"""

# ============================================================
# 1. 列表推导式（进阶）
# ============================================================

# 嵌套推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 展平二维列表
flat = [x for row in matrix for x in row]
print(f"展平: {flat}")

# 矩阵转置
transposed = [[row[i] for row in matrix] for i in range(3)]
print(f"转置: {transposed}")

# 字典推导式
adj = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
degree_dict = {node: len(neighbors) for node, neighbors in adj.items()}
print(f"度字典: {degree_dict}")

# 集合推导式
all_neighbors = {n for neighbors in adj.values() for n in neighbors}
print(f"所有邻居: {all_neighbors}")

# ============================================================
# 2. 生成器表达式和生成器函数
# ============================================================

# 生成器表达式（圆括号）
sum_of_squares = sum(x**2 for x in range(1000000))
print(f"\n大数求和: {sum_of_squares}")

# 生成器管道（链式处理）
def read_edges():
    """模拟读取边"""
    for i in range(10):
        yield (i, i+1)

def filter_edges(edges, min_node):
    """过滤边"""
    for u, v in edges:
        if u >= min_node and v >= min_node:
            yield (u, v)

def add_weights(edges):
    """给边加权重"""
    for u, v in edges:
        yield (u, v, 1.0)

# 管道：读取 -> 过滤 -> 加权重
pipeline = add_weights(filter_edges(read_edges(), 3))
print("\n生成器管道:")
for edge in pipeline:
    print(f"  {edge}")

# ============================================================
# 3. 装饰器（进阶）
# ============================================================

import random
import functools
import time

# 带参数的装饰器
def retry(max_attempts=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)  # 保留原函数的元信息
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"  重试 {attempt + 1}/{max_attempts}: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unreliable_network_analysis(n):
    """模拟不稳定的网络分析"""
    if random.random() < 0.3:
        raise ValueError("分析失败")
    return n * 10

random.seed(42)

print("\n重试装饰器:")
try:
    result = unreliable_network_analysis(100)
    print(f"  结果: {result}")
except ValueError as e:
    print(f"  最终失败: {e}")

# ============================================================
# 4. 上下文管理器（with语句）
# ============================================================

from contextlib import contextmanager

class Timer:
    """计时上下文管理器"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        print(f"  耗时: {self.elapsed:.4f}秒")
        return False  # 不抑制异常

# 使用with语句
print("\n上下文管理器:")
with Timer() as t:
    time.sleep(0.1)
    total = sum(x**2 for x in range(100000))

# 更优雅的方式：用contextmanager装饰器
@contextmanager
def network_temp_dir(path):
    """临时目录上下文管理器"""
    import os
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        # 清理（这里只是演示，实际不会删除）
        print(f"  清理临时目录: {path}")

with network_temp_dir("E:/ai_python/data/temp") as tmp_dir:
    print(f"  使用临时目录: {tmp_dir}")

# ============================================================
# 5. 迭代器协议
# ============================================================

class DegreeIterator:
    """迭代器：遍历网络的度序列"""

    def __init__(self, adjacency_list):
        self.adj = adjacency_list
        self.nodes = sorted(adjacency_list.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.nodes):
            raise StopIteration
        node = self.nodes[self.index]
        degree = len(self.adj[node])
        self.index += 1
        return node, degree

# 使用迭代器
adj = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1, 4], 4: [3]}
print("\n迭代器:")
for node, degree in DegreeIterator(adj):
    print(f"  节点{node}: 度={degree}")

# ============================================================
# 6. 异步编程基础（async/await）
# ============================================================

import asyncio

async def simulate_propagation(network_name, steps):
    """模拟传播过程（异步）"""
    print(f"  [{network_name}] 开始传播")
    for step in range(steps):
        await asyncio.sleep(0.01)  # 模拟异步操作
        print(f"  [{network_name}] 步骤 {step + 1}")
    print(f"  [{network_name}] 传播完成")

async def run_simulations():
    """并行运行多个仿真"""
    print("\n异步并行仿真:")
    await asyncio.gather(
        simulate_propagation("ER网络", 3),
        simulate_propagation("BA网络", 3),
    )

# 运行异步代码
# asyncio.run(run_simulations())  # 在实际代码中取消注释

# ============================================================
# 7. 类型提示（Type Hints）- 提高代码可读性
# ============================================================

from typing import List, Dict, Tuple, Optional

def analyze_network(
    adjacency: Dict[int, List[int]],
    directed: bool = False
) -> Dict[str, float]:
    """类型提示的函数"""
    n = len(adjacency)
    total_degree = sum(len(neighbors) for neighbors in adjacency.values())
    avg_degree = total_degree / n if n > 0 else 0

    # 集聚系数（简化版）
    clustering = 0.0
    for node, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        triangles = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adjacency[neighbors[i]]:
                    triangles += 1
        k = len(neighbors)
        clustering += 2 * triangles / (k * (k - 1))
    clustering /= n

    return {
        "num_nodes": n,
        "avg_degree": avg_degree,
        "clustering": clustering
    }

# 使用
result = analyze_network(adj)
print(f"\n类型提示: {result}")

# ============================================================
# 8. 内置工具函数
# ============================================================

print("\n常用内置工具:")

# itertools - 迭代器工具
import itertools

# 组合（网络中枚举子图）
nodes = [0, 1, 2, 3, 4]
triangles = list(itertools.combinations(nodes, 3))
print(f"  三角形组合: {triangles[:5]}...")

# 排列
edges_possible = list(itertools.permutations(nodes, 2))
print(f"  所有可能边数: {len(edges_possible)}")

# chain - 连接多个迭代器
list1 = [1, 2, 3]
list2 = [4, 5, 6]
chained = list(itertools.chain(list1, list2))
print(f"  链接: {chained}")

# functools - 函数工具
from functools import reduce

# 用reduce计算连乘
product = reduce(lambda x, y: x * y, range(1, 6))
print(f"  5! = {product}")

# lru_cache - 缓存函数结果（记忆化）
@functools.lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"  Fibonacci(20) = {fibonacci(20)}")

# ============================================================
# 练习：实现一个惰性网络加载器
# ============================================================

def exercise_lazy_loader():
    """实现一个惰性加载大型网络的生成器"""
    import csv
    from pathlib import Path

    def lazy_load_edges(filepath):
        """惰性加载边列表（生成器）"""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield (int(row['source']), int(row['target']),
                       float(row.get('weight', 1.0)))

    # 使用
    data_path = Path("E:/ai_python/data/edges.csv")
    if data_path.exists():
        print("\n惰性加载边:")
        edge_gen = lazy_load_edges(data_path)
        for edge in edge_gen:
            print(f"  {edge}")

if __name__ == "__main__":
    exercise_lazy_loader()
