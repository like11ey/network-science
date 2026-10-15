#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/phase6_transition_order.py — 用**细 f 网格 + 网格无关判据**给 L1/L2 定相变阶。

为什么单独写这个脚本？
    L0 的实验用的是 46 点网格，`transition_order` 的"跳变"判据依赖网格密度：
    网格粗时把连续下降看成小台阶，网格细时又能凑出一个大跳变。README 里已记下这个坑。
    本脚本用**同一批网络 + 同一批初始故障**扫细网格（默认 0.02 步长），
    并同时报三种判据（最大单步跌幅 / max|dS/df| / 二阶中点的 p_c），
    免得"一阶"结论只是网格凑出来的。

用法（项目根目录）：
    python scripts/phase6_transition_order.py --level L1 --N 1000 --trials 5
    python scripts/phase6_transition_order.py --level L2 --N 1000 --trials 5
"""

import argparse
import importlib
import json
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

core = importlib.import_module("phase6_interdependent.core")
L1 = importlib.import_module("phase6_interdependent.L1_intra_layer_dynamics")
L2 = importlib.import_module("phase6_interdependent.L2_state_driven_coupling")
metrics = importlib.import_module("metrics.resilience_metrics")

RESULTS = os.path.join(project_root, "results")


def er_pair(N, k_avg=4.0, seed=1):
    """两层 ER（⟨k⟩ 默认 4，对齐 PNAS 2019 图 3 的口径）。"""
    p = k_avg / (N - 1)
    return (nx.erdos_renyi_graph(N, p, seed=seed),
            nx.erdos_renyi_graph(N, p, seed=seed + 1))


def curve_for(make_rule, G_A, G_B, fractions, q, n_trials, seed):
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())
    curve = []
    for f in fractions:
        n_remove = int(round(len(nodes_A) * f))
        vals = []
        for _ in range(n_trials):
            net = core.InterdependentNetwork(G_A, G_B, coupled_fraction=q, seed=seed)
            failed = ([] if n_remove == 0
                      else rng.choice(nodes_A, size=n_remove, replace=False))
            vals.append(net.run(failed, rule=make_rule(net))["S_A"])
        curve.append(float(np.mean(vals)))
    return np.array(curve)


def make_L1_rule(delta, q, G_A, G_B, seed):
    cfg = L1.AutoDynamicsConfig(mu=0.4, lam=0.3, omega=0.3, delta=delta)
    x_ref = L1.healthy_reference(G_A, G_B, cfg, coupled_fraction=q, seed=seed)
    return lambda net: L1.L1Rule(cfg, x_ref)


def make_L2_rule(rho, phi, q):
    cfg = L2.StateDrivenConfig(w0=1.0, rho=rho, phi=phi, omega=0.3)
    return lambda net: L2.L2Rule(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L1", "L2"], default="L1")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--fmax", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", default="")
    args = ap.parse_args()

    fractions = np.arange(0.0, args.fmax + 1e-9, args.step)
    G_A, G_B = er_pair(args.N, seed=args.seed)
    print(f"ER 双层 ⟨k⟩≈4, N={args.N}, 网格 {len(fractions)} 点, {args.trials} 次重复, seed={args.seed}")

    if args.level == "L1":
        variants = ([float(x) for x in args.variants.split(",")] if args.variants
                    else [0.05, 0.2, 0.5, 1.0])   # δ；δ=1.0 是 L0 退化基线
        rule_of = lambda v, q: make_L1_rule(v, q, G_A, G_B, args.seed)
        label = lambda v: f"δ={v}"
    else:
        variants = ([tuple(float(y) for y in x.split(":")) for x in args.variants.split(",")]
                    if args.variants else [(0.0, 0.0), (1.0, 0.3)])
        rule_of = lambda v, q: make_L2_rule(v[0], v[1], q)
        label = lambda v: f"ρ={v[0]},φ={v[1]}"

    report = {}
    table = []
    for q in (0.2, 1.0):
        for v in variants:
            t0 = time.time()
            S = curve_for(lambda net, v=v, q=q: rule_of(v, q)(net),
                          G_A, G_B, fractions, q, args.trials, args.seed)
            order, jump = metrics.transition_order(fractions, S, jump_threshold=0.2)
            row = {
                "q": q, "variant": str(v), "order": order, "max_jump": round(jump, 4),
                "max_slope": round(metrics.max_slope(fractions, S), 4),
                "pc_mid": round(metrics.threshold_by_midpoint(fractions, S), 4),
                "pc_drop": round(metrics.threshold_by_max_drop(fractions, S), 4),
                "S_f02": round(float(np.interp(0.2, fractions, S)), 4),
                "S_f06": round(float(np.interp(0.6, fractions, S)), 4),
                "loss_area": round(metrics.loss_area(fractions, S), 4),
                "sec": round(time.time() - t0, 1),
            }
            table.append(row)
            report[f"q={q}|{label(v)}"] = list(map(float, S))
            print(f"  q={q} {label(v):<14} → {order:<3} 最大单步={jump:.3f} "
                  f"max_slope={row['max_slope']:.3f} p_c≈{row['pc_drop']:.2f} "
                  f"S(0.2)={row['S_f02']:.3f} S(0.6)={row['S_f06']:.3f} ({row['sec']}s)")

    print(f"\n=== {args.level} 相变阶判定（ER ⟨k⟩≈4, N={args.N}）===")
    print(f"{'q':>4} {'变体':>14} {'判阶':>4} {'最大单步':>8} {'max_slope':>10} {'p_c(跌)':>8}")
    for r in table:
        print(f"{r['q']:>4} {r['variant']:>14} {r['order']:>4} {r['max_jump']:>8.3f} "
              f"{r['max_slope']:>10.3f} {r['pc_drop']:>8.2f}")
    print("判读：max_slope 是网格无关的'崩得多猛'；q=1.0 应显著大于 q=0.2，才说明强耦合更脆。")

    os.makedirs(RESULTS, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for key, S in report.items():
        q = float(key.split("|")[0].split("=")[1])
        plt.plot(fractions, S, "o-", markersize=3, alpha=0.9 if q == 1.0 else 0.5,
                 linestyle="-" if q == 1.0 else "--", label=key)
    plt.xlabel("初始移除比例 f")
    plt.ylabel("存活序参量 S")
    plt.title(f"{args.level} 相变阶（ER ⟨k⟩≈4, N={args.N}, {args.trials} 次重复）")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    out = os.path.join(RESULTS, f"phase6_{args.level}_transition_order.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    with open(os.path.join(RESULTS, f"phase6_{args.level}_transition_order.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"fractions": list(fractions), "table": table, "curves": report},
                  fh, ensure_ascii=False, indent=2)
    print(f"图/数据已保存：{out}")


if __name__ == "__main__":
    main()
