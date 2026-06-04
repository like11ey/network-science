"""
Phase 1.2 - 条件语句与循环
对比C/Java，Python用缩进代替花括号
"""

# ============================================================
# 1. 条件语句 - 注意：没有花括号，用缩进
# ============================================================

# C/Java: if (x > 0) { ... } else if (x == 0) { ... }
# Python:
x = 42

if x > 0:
    print("正数")
elif x == 0:  # 注意是 elif，不是 else if
    print("零")
else:
    print("负数")

# 多条件组合 - Python可以用 and/or 直接连接（不需要 &&/||）
k = 5
if 2 <= k <= 10:  # Python支持链式比较！C/Java做不到
    print(f"k={k} 在 [2,10] 范围内")

# ============================================================
# 2. for循环 - Python的for是for-each风格
# ============================================================

# C: for (int i = 0; i < 5; i++) { ... }
# Python:
print("\n--- range() 循环 ---")
for i in range(5):  # 0, 1, 2, 3, 4
    print(i, end=" ")
print()

# 带步长
print("\n--- 带步长 ---")
for i in range(0, 20, 3):  # 0, 3, 6, 9, 12, 15, 18
    print(i, end=" ")
print()

# 遍历列表
print("\n--- 遍历列表 ---")
models = ["ER随机", "WS小世界", "BA无标度", "规则网络"]
for model in models:
    print(f"  网络模型: {model}")

# 带索引遍历（用 enumerate）
print("\n--- enumerate ---")
for i, model in enumerate(models):
    print(f"  {i+1}. {model}")

# ============================================================
# 3. while循环
# ============================================================

print("\n--- while循环 ---")
count = 0
while count < 5:
    print(count, end=" ")
    count += 1  # 没有 ++ 运算符
print()

# ============================================================
# 4. 列表推导式（List Comprehension）- Python最强大的特性之一
# ============================================================

# C/Java需要写循环再一个个append
# Python一行搞定：
squares = [x**2 for x in range(10)]
print(f"\n平方数: {squares}")

# 带条件的推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"偶数的平方: {even_squares}")

# 实际应用：计算网络的度序列
import random
degrees = [random.randint(1, 20) for _ in range(100)]  # 模拟100个节点的度
print(f"\n度序列前10个: {degrees[:10]}")

# 用推导式统计度分布
from collections import Counter
degree_dist = Counter(degrees)
print(f"度分布: {dict(sorted(degree_dist.items()))}")

# ============================================================
# 5. 三元表达式（条件表达式）
# ============================================================

# C: int max = (a > b) ? a : b;
# Python:
a, b = 10, 20
max_val = a if a > b else b
print(f"\n最大值: {max_val}")

# ============================================================
# 6. 字典映射（Python 3.9中替代match-case的方式）
# ============================================================

print("\n--- 字典映射（等效switch-case）---")
network_type = "scale_free"

network_info = {
    "random": "ER随机网络 - Poisson度分布",
    "small_world": "WS小世界网络 - 高集聚系数",
    "scale_free": "BA无标度网络 - 幂律度分布",
}
# get的第二个参数是默认值（类似default分支）
print(f"  {network_info.get(network_type, '未知网络类型')}")

# ============================================================
# 7. 异常处理的try-except（预告Phase 1.8）
# ============================================================

# Python的异常处理和C/Java类似，但更简洁
numbers = [1, 2, 3]
try:
    value = numbers[10]
except IndexError as e:
    print(f"\n索引越界: {e}")
except Exception as e:
    print(f"其他错误: {e}")
finally:
    print("finally块始终执行")

# ============================================================
# 练习：生成BA无标度网络的度序列（简化版）
# ============================================================

def exercise_ba_degree_sequence():
    """
    模拟BA模型的度分布生成过程
    提示：使用列表推导式
    """
    import random
    random.seed(42)

    N = 1000  # 节点数
    m = 3     # 每个新节点连接的边数

    # 用列表推导式生成初始m个节点的度
    degrees = [m] * m

    # 模拟增长过程（简化版）
    for i in range(m, N):
        # 每个新节点带来m条边，每条边增加一个节点的度
        for _ in range(m):
            # 从现有节点中随机选择（偏好连接的简化）
            target = random.choices(range(len(degrees)), weights=degrees)[0]
            degrees[target] += 1
        degrees.append(m)

    # 计算度分布
    degree_dist = Counter(degrees)
    print(f"节点数: {N}")
    print(f"最大度: {max(degrees)}")
    print(f"平均度: {sum(degrees)/len(degrees):.2f}")
    print(f"度分布(前10): {dict(list(sorted(degree_dist.items()))[:10])}")

if __name__ == "__main__":
    exercise_ba_degree_sequence()
