"""
Phase 1.6 - 面向对象编程
Python的OOP比C/Java更灵活（多继承、元类等）
"""

# ============================================================
# 1. 基本类定义
# ============================================================

class Network:
    """网络基类"""

    # 类属性（所有实例共享）
    MAX_NODES = 10000

    def __init__(self, name: str, num_nodes: int):
        """构造函数（Java的constructor）"""
        # 实例属性（每个实例独立）
        self.name = name
        self.num_nodes = num_nodes
        self.edges = []

    def add_edge(self, u: int, v: int):
        """添加边"""
        if u >= self.num_nodes or v >= self.num_nodes:
            raise ValueError(f"节点编号超出范围: {u} 或 {v}")
        self.edges.append((u, v))

    def num_edges(self) -> int:
        """返回边数"""
        return len(self.edges)

    def avg_degree(self) -> float:
        """计算平均度"""
        return 2 * len(self.edges) / self.num_nodes if self.num_nodes > 0 else 0

    def __repr__(self):
        """打印时的表示（类似Java的toString）"""
        return f"Network('{self.name}', nodes={self.num_nodes}, edges={self.num_edges()})"

# 使用
net = Network("测试网络", 5)
net.add_edge(0, 1)
net.add_edge(1, 2)
net.add_edge(2, 3)
print(net)
print(f"平均度: {net.avg_degree():.2f}")

# ============================================================
# 2. 继承
# ============================================================

import random

class RandomNetwork(Network):
    """ER随机网络"""

    def __init__(self, num_nodes: int, p: float):
        super().__init__(f"ER(p={p})", num_nodes)  # 调用父类构造函数
        self.p = p
        self._generate()  # 自动生成网络

    def _generate(self):
        """生成随机网络"""
        random.seed(42)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < self.p:
                    self.add_edge(i, j)

class ScaleFreeNetwork(Network):
    """BA无标度网络"""

    def __init__(self, num_nodes: int, m: int):
        super().__init__(f"BA(m={m})", num_nodes)
        self.m = m
        self._generate()

    def _generate(self):
        """生成无标度网络"""
        random.seed(42)
        # 初始完全连接
        for i in range(self.m):
            for j in range(i + 1, self.m):
                self.add_edge(i, j)

        # 度数跟踪
        degrees = [0] * self.num_nodes
        for u, v in self.edges:
            degrees[u] += 1
            degrees[v] += 1

        # 增长过程
        for new_node in range(self.m, self.num_nodes):
            # 偏好连接：按度数比例选择目标
            total_degree = sum(degrees[:new_node])
            if total_degree == 0:
                targets = random.sample(range(new_node), self.m)
            else:
                probs = [d / total_degree for d in degrees[:new_node]]
                targets = []
                available = list(range(new_node))
                for _ in range(self.m):
                    target = random.choices(available, weights=probs[:new_node])[0]
                    targets.append(target)
                    probs[target] = 0  # 避免重复选择

            for t in targets:
                self.add_edge(new_node, t)
                degrees[new_node] += 1
                degrees[t] += 1

# 使用继承
er_net = RandomNetwork(100, 0.05)
print(f"\nER网络: {er_net}")
print(f"  平均度: {er_net.avg_degree():.2f}")

ba_net = ScaleFreeNetwork(100, 3)
print(f"\nBA网络: {ba_net}")
print(f"  平均度: {ba_net.avg_degree():.2f}")

# ============================================================
# 3. 多态（Python天然支持）
# ============================================================

def analyze_network(net: Network):
    """多态：分析任何类型的网络"""
    print(f"\n分析 {net.name}:")
    print(f"  节点数: {net.num_nodes}")
    print(f"  边数: {net.num_edges()}")
    print(f"  平均度: {net.avg_degree():.2f}")

analyze_network(er_net)
analyze_network(ba_net)

# ============================================================
# 4. 属性装饰器（@property）
# ============================================================

class WeightedEdge:
    """带权重的边"""

    def __init__(self, u, v, weight=1.0):
        self._u = u
        self._v = v
        self._weight = weight

    @property
    def weight(self):
        """getter - 像访问属性一样调用"""
        return self._weight

    @weight.setter
    def weight(self, value):
        """setter - 像赋值一样调用"""
        if value < 0:
            raise ValueError("权重不能为负")
        self._weight = value

    @property
    def endpoints(self):
        """只读属性"""
        return (self._u, self._v)

# 使用
edge = WeightedEdge(0, 1, 2.5)
print(f"\n边: {edge.endpoints}, 权重: {edge.weight}")
edge.weight = 3.0  # 像属性一样赋值
print(f"新权重: {edge.weight}")

# ============================================================
# 5. 魔术方法（Dunder Methods）
# ============================================================

class Node:
    """节点类，展示常用魔术方法"""

    def __init__(self, id, label=None):
        self.id = id
        self.label = label or f"Node_{id}"
        self.degree = 0

    def __eq__(self, other):
        """相等比较 =="""
        return self.id == other.id

    def __lt__(self, other):
        """小于比较 <（用于排序）"""
        return self.id < other.id

    def __hash__(self):
        """哈希值（用于集合和字典的key）"""
        return hash(self.id)

    def __repr__(self):
        """repr（调试用）"""
        return f"Node(id={self.id}, label='{self.label}')"

    def __str__(str):
        """str（用户友好）"""
        return f"{str.label}(deg={str.degree})"

# 使用
nodes = [Node(2, "C"), Node(0, "A"), Node(1, "B")]
print(f"\n排序前: {nodes}")
print(f"排序后: {sorted(nodes)}")

# ============================================================
# 6. 数据类（dataclass）- Python 3.7+ 简化类定义
# ============================================================

from dataclasses import dataclass, field

@dataclass
class NetworkMetrics:
    """网络指标数据类"""
    num_nodes: int
    num_edges: int
    avg_degree: float
    clustering: float = 0.0
    avg_path_length: float = float('inf')

    @property
    def density(self):
        """网络密度"""
        if self.num_nodes <= 1:
            return 0
        max_edges = self.num_nodes * (self.num_nodes - 1) / 2
        return self.num_edges / max_edges

# 使用 - 自动获得__init__, __repr__, __eq__等方法
metrics = NetworkMetrics(1000, 5000, 10.0, 0.35, 3.2)
print(f"\n数据类: {metrics}")
print(f"密度: {metrics.density:.4f}")

# ============================================================
# 7. 组合优于继承
# ============================================================

class NetworkAnalyzer:
    """通过组合使用不同策略"""

    def __init__(self, network: Network):
        self.network = network
        self._degree_cache = None

    def get_degree_sequence(self):
        """获取度序列"""
        if self._degree_cache is None:
            from collections import Counter
            adj = {}
            for u, v in self.network.edges:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            self._degree_cache = [len(adj.get(i, [])) for i in range(self.network.num_nodes)]
        return self._degree_cache

    def get_degree_distribution(self):
        """获取度分布"""
        from collections import Counter
        return dict(Counter(self.get_degree_sequence()))

# 使用
analyzer = NetworkAnalyzer(ba_net)
print(f"\nBA网络度分布: {analyzer.get_degree_distribution()}")

# ============================================================
# 练习：设计一个完整的网络类层次
# ============================================================

def exercise_class_hierarchy():
    """
    设计类层次：BaseGraph -> ErdosRenyi / WattsStrogatz / BarabasiAlbert
    每个类实现 generate() 和 get_degree_distribution()
    """
    import random
    random.seed(42)

    class BaseGraph:
        def __init__(self, n):
            self.n = n
            self.edges = []
            self.adj = {i: [] for i in range(n)}

        def add_edge(self, u, v):
            self.edges.append((u, v))
            self.adj[u].append(v)
            self.adj[v].append(u)

        def degree_distribution(self):
            from collections import Counter
            degrees = [len(self.adj[i]) for i in range(self.n)]
            return dict(Counter(degrees))

    class ErdosRenyi(BaseGraph):
        def __init__(self, n, p):
            super().__init__(n)
            self.p = p
            self._generate()

        def _generate(self):
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if random.random() < self.p:
                        self.add_edge(i, j)

    class BarabasiAlbert(BaseGraph):
        def __init__(self, n, m):
            super().__init__(n)
            self.m = m
            self._generate()

        def _generate(self):
            # 初始完全连接
            for i in range(self.m):
                for j in range(i+1, self.m):
                    self.add_edge(i, j)

            degrees = [len(self.adj[i]) for i in range(self.n)]
            for new_node in range(self.m, self.n):
                total = sum(degrees[:new_node])
                probs = [d/total if total > 0 else 1/new_node for d in degrees[:new_node]]
                targets = random.choices(range(new_node), weights=probs, k=self.m)
                for t in targets:
                    self.add_edge(new_node, t)
                    degrees[new_node] += 1
                    degrees[t] += 1

    # 测试
    er = ErdosRenyi(500, 0.02)
    ba = BarabasiAlbert(500, 3)

    print("\n类层次练习:")
    print(f"ER度分布: {er.degree_distribution()}")
    print(f"BA度分布: {ba.degree_distribution()}")

if __name__ == "__main__":
    exercise_class_hierarchy()
