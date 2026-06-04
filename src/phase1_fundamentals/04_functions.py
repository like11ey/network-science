"""
Phase 1.4 - 函数、lambda、闭包、装饰器
Python函数是一等公民（可以赋值、传递、返回）
"""

# ============================================================
# 1. 基本函数定义
# ============================================================

# C: double calculate_avg_degree(int n, double p) { ... }
# Python:
def calculate_avg_degree(n: int, p: float) -> float:
    """计算ER随机网络的平均度（类型注解是可选的，用于提高可读性）"""
    return n * p

result = calculate_avg_degree(1000, 0.01)
print(f"平均度: {result}")

# 默认参数 - C/Java不支持
def create_edge(u, v, weight=1.0):
    """创建边，默认权重为1.0"""
    return (u, v, weight)

print(f"默认权重: {create_edge(0, 1)}")
print(f"指定权重: {create_edge(0, 1, 2.5)}")

# 关键字参数 - 调用时可以指定参数名
print(f"关键字参数: {create_edge(u=0, v=1, weight=3.0)}")

# ============================================================
# 2. 可变参数
# ============================================================

# *args - 接收任意数量的位置参数（元组）
def calculate_total_degree(*degrees):
    """计算所有节点的度之和"""
    return sum(degrees)

print(f"\n度之和: {calculate_total_degree(1, 2, 3, 4, 5)}")

# **kwargs - 接收任意数量的关键字参数（字典）
def create_network_info(name="未知", **kwargs):
    """创建网络信息"""
    info = {"name": name}
    info.update(kwargs)
    return info

info = create_network_info(
    name="ER网络",
    nodes=1000,
    edges=5000,
    avg_degree=10.0
)
print(f"网络信息: {info}")

# ============================================================
# 3. Lambda函数（匿名函数）
# ============================================================

# 普通函数
def square(x):
    return x ** 2

# 等价的lambda
square_lambda = lambda x: x ** 2
print(f"\nLambda: {square_lambda(5)}")

# Lambda最常用于排序key
nodes = [(0, 3), (1, 5), (2, 1), (3, 4), (4, 2)]
# 按度值（第二个元素）排序
sorted_by_degree = sorted(nodes, key=lambda node: node[1])
print(f"按度排序: {sorted_by_degree}")

# 按度值降序
sorted_desc = sorted(nodes, key=lambda node: -node[1])
print(f"按度降序: {sorted_desc}")

# ============================================================
# 4. 闭包（Closure）- 函数返回函数
# ============================================================

def create_degree_counter(initial_count=0):
    """创建一个度计数器（闭包）"""
    count = initial_count

    def increment():
        nonlocal count  # 声明使用外层变量
        count += 1
        return count

    def get_count():
        return count

    return increment, get_count

inc, get = create_degree_counter()
print(f"\n闭包测试:")
print(f"  第一次调用: {inc()}")  # 1
print(f"  第二次调用: {inc()}")  # 2
print(f"  获取计数: {get()}")    # 2

# ============================================================
# 5. 装饰器（Decorator）- Python独有的强大特性
# ============================================================

import time

def timer(func):
    """计时装饰器 - 测量函数执行时间"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"  {func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

def log_calls(func):
    """日志装饰器 - 记录函数调用"""
    def wrapper(*args, **kwargs):
        print(f"  调用 {func.__name__}(args={args}, kwargs={kwargs})")
        result = func(*args, **kwargs)
        print(f"  返回: {result}")
        return result
    return wrapper

# 使用装饰器 - 用 @ 语法糖
@timer
def slow_network_analysis(n):
    """模拟一个耗时的网络分析"""
    time.sleep(0.1)  # 模拟耗时操作
    return n * 10

@log_calls
def simple_calc(x, y):
    return x + y

print("\n装饰器测试:")
result = slow_network_analysis(100)
simple_calc(3, 4)

# ============================================================
# 6. 生成器（Generator）- 惰性计算，节省内存
# ============================================================

# 普通函数 - 一次性返回所有结果
def get_all_nodes(n):
    nodes = []
    for i in range(n):
        nodes.append(i)
    return nodes

# 生成器函数 - 逐个yield结果
def get_nodes_generator(n):
    for i in range(n):
        yield i  # yield代替return

print("\n生成器测试:")
# 普通函数
nodes_list = get_all_nodes(5)
print(f"列表: {nodes_list}")

# 生成器
gen = get_nodes_generator(5)
print(f"生成器: {gen}")  # 不会立即执行
print(f"next: {next(gen)}")  # 0
print(f"next: {next(gen)}")  # 1

# 生成器最常用于for循环
print("生成器遍历:")
for node in get_nodes_generator(5):
    print(f"  节点 {node}")

# 实际应用：生成边列表（节省内存）
def generate_edges(n, m):
    """生成BA模型的边（简化版）"""
    import random
    random.seed(42)

    # 初始m个节点的完全连接
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((i, j))
            yield (i, j)

    # 模拟增长过程
    for new_node in range(m, n):
        targets = random.sample(range(new_node), m)
        for t in targets:
            edges.append((new_node, t))
            yield (new_node, t)

print("\n生成边（惰性）:")
edge_gen = generate_edges(6, 2)
for edge in edge_gen:
    print(f"  {edge}")

# ============================================================
# 7. map, filter, reduce
# ============================================================

from functools import reduce

degrees = [1, 2, 3, 4, 5]

# map: 对每个元素应用函数
squared = list(map(lambda x: x**2, degrees))
print(f"\nmap平方: {squared}")

# filter: 筛选元素
evens = list(filter(lambda x: x % 2 == 0, degrees))
print(f"filter偶数: {evens}")

# reduce: 累积计算
product = reduce(lambda x, y: x * y, degrees)
print(f"reduce乘积: {product}")

# 更Pythonic的方式（用列表推导式替代map和filter）
squared_p = [x**2 for x in degrees]
evens_p = [x for x in degrees if x % 2 == 0]
print(f"推导式平方: {squared_p}")
print(f"推导式偶数: {evens_p}")

# ============================================================
# 练习：实现一个网络度分布分析器
# ============================================================

def exercise_degree_analyzer():
    """综合练习：使用函数、lambda、列表推导式分析度分布"""
    import random
    random.seed(42)

    # 模拟1000个节点的度
    degrees = [random.randint(1, 30) for _ in range(1000)]

    # 1. 用Counter统计度分布
    from collections import Counter
    dist = Counter(degrees)

    # 2. 用sorted和lambda排序
    sorted_dist = sorted(dist.items(), key=lambda x: -x[1])  # 按频次降序
    print("\n度分布（按频次降序）:")
    for deg, freq in sorted_dist[:10]:
        print(f"  度={deg}: {freq}次 ({freq/len(degrees)*100:.1f}%)")

    # 3. 用reduce计算总边数
    total_degree = reduce(lambda x, y: x + y, degrees)
    print(f"\n总度数: {total_degree}")
    print(f"总边数: {total_degree // 2}")
    print(f"平均度: {total_degree / len(degrees):.2f}")

if __name__ == "__main__":
    exercise_degree_analyzer()
