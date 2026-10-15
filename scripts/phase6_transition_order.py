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
    python scripts/phase6_transition_order.py --level L3 --N 500 --trials 2 --step 0.05
        # L3 默认对比三种功能单元语义：AND / k-of-N / OR（超边 size 由 --size 控制）
    python scripts/phase6_transition_order.py --level CPS --N 300 --trials 2 --step 0.05
        # CPS 主循环：变体写法 = lambda_tau:alpha:beta:delta:phi_s（如 0:1:0:1:0、1:1:0.5:0.3:0.75）
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
L3 = importlib.import_module("phase6_interdependent.L3_hypergraph_logic")
CPS = importlib.import_module("phase6_interdependent.cps_cascade")
metrics = importlib.import_module("metrics.resilience_metrics")

RESULTS = os.path.join(project_root, "results")


def er_pair(N, k_avg=4.0, seed=1):
    """两层 ER（⟨k⟩ 默认 4，对齐 PNAS 2019 图 3 的口径）。"""
    p = k_avg / (N - 1)
    return (nx.erdos_renyi_graph(N, p, seed=seed),
            nx.erdos_renyi_graph(N, p, seed=seed + 1))


def default_factory(G_A, G_B, q, seed):
    return core.InterdependentNetwork(G_A, G_B, coupled_fraction=q, seed=seed)


def curve_for(make_rule, G_A, G_B, fractions, q, n_trials, seed,
              net_factory=None, measure=None):
    """同一批网络 + 同一批初始故障扫 f；可选额外度量（L3 的功能度量 F）。"""
    factory = net_factory or default_factory
    rng = np.random.default_rng(seed)
    nodes_A = sorted(G_A.nodes())
    curve, extra = [], []
    for f in fractions:
        n_remove = int(round(len(nodes_A) * f))
        vals, extras = [], []
        for _ in range(n_trials):
            net = factory(G_A, G_B, q, seed)
            failed = ([] if n_remove == 0
                      else rng.choice(nodes_A, size=n_remove, replace=False))
            net.run(failed, rule=make_rule(net))
            vals.append(net.S("A"))
            if measure is not None:
                extras.append(measure(net))
        curve.append(float(np.mean(vals)))
        if measure is not None:
            extra.append(float(np.mean(extras)))
    return np.array(curve), (np.array(extra) if measure is not None else None)


def make_L1_rule(delta, q, G_A, G_B, seed):
    cfg = L1.AutoDynamicsConfig(mu=0.4, lam=0.3, omega=0.3, delta=delta)
    x_ref = L1.healthy_reference(G_A, G_B, cfg, coupled_fraction=q, seed=seed)
    return lambda net: L1.L1Rule(cfg, x_ref)


def make_L2_rule(rho, phi, q):
    cfg = L2.StateDrivenConfig(w0=1.0, rho=rho, phi=phi, omega=0.3)
    return lambda net: L2.L2Rule(cfg)


def make_L3_factory(rule, k, n_edges, size, seed):
    """返回"带同一批功能单元的超图网络"工厂：同一批超边才能公平对比规则语义。"""
    def factory(G_A, G_B, q, net_seed):
        template = L3.HyperedgeNetwork(G_A, G_B, hyperedges=[], seed=net_seed)
        edges = L3.random_functional_hyperedges(template, n_edges=n_edges, size=size,
                                                rule=rule, k=k, seed=seed)
        return L3.HyperedgeNetwork(G_A, G_B, hyperedges=edges,
                                   coupled_fraction=q, seed=net_seed)
    return factory


def make_CPS_factory(n_edges, size, seed):
    """CPS 主循环：同一批 k-of-N 功能单元，层间耦合由 cfg 的 W^C/W^S 控制。"""
    def factory(G_A, G_B, q, net_seed):
        template = L3.HyperedgeNetwork(G_A, G_B, hyperedges=[], seed=net_seed)
        edges = L3.random_functional_hyperedges(template, n_edges=n_edges, size=size,
                                                rule="k-of-N", k=2, seed=seed)
        return CPS.CPSNetwork(G_A, G_B, hyperedges=edges,
                              coupled_fraction=q, seed=net_seed)
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L1", "L2", "L3", "CPS"], default="L1")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--fmax", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", default="")
    ap.add_argument("--edges", type=int, default=60, help="L3：功能单元（超边）条数")
    ap.add_argument("--size", type=int, default=3, help="L3：每条超边的成员数")
    args = ap.parse_args()

    fractions = np.arange(0.0, args.fmax + 1e-9, args.step)
    G_A, G_B = er_pair(args.N, seed=args.seed)
    print(f"ER 双层 ⟨k⟩≈4, N={args.N}, 网格 {len(fractions)} 点, {args.trials} 次重复, seed={args.seed}")

    if args.level == "L1":
        variants = ([float(x) for x in args.variants.split(",")] if args.variants
                    else [0.05, 0.2, 0.5, 1.0])   # δ；δ=1.0 是 L0 退化基线
        rule_of = lambda v, q: make_L1_rule(v, q, G_A, G_B, args.seed)
        net_factory_of = lambda v: None
        measure = None
        label = lambda v: f"δ={v}"
    elif args.level == "L2":
        variants = ([tuple(float(y) for y in x.split(":")) for x in args.variants.split(",")]
                    if args.variants else [(0.0, 0.0), (1.0, 0.3)])
        rule_of = lambda v, q: make_L2_rule(v[0], v[1], q)
        net_factory_of = lambda v: None
        measure = None
        label = lambda v: f"ρ={v[0]},φ={v[1]}"
    elif args.level == "CPS":
        variants = ([tuple(float(y) for y in x.split(":")) for x in args.variants.split(",")]
                    if args.variants
                    else [(0.0, 1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.3, 0.0),
                          (0.0, 1.0, 0.0, 1.0, 0.75), (1.0, 1.0, 0.5, 0.3, 0.75)])
        cfgs = {v: CPS.CPSConfig(lambda_tau=v[0], alpha=v[1], beta=v[2],
                                 delta=v[3], phi_s=v[4]) for v in variants}
        x_refs = {(v, q): CPS.healthy_reference(G_A, G_B, cfgs[v], coupled_fraction=q,
                                                seed=args.seed)
                  for v in variants for q in (0.2, 1.0)}
        rule_of = lambda v, q: (lambda net: net.cps_rule(cfgs[v], x_refs[(v, q)]))
        net_factory_of = lambda v: make_CPS_factory(args.edges, args.size, args.seed)
        measure = L3.functional_measure
        label = lambda v: f"λτ={v[0]}/α={v[1]}/β={v[2]}/δ={v[3]}/φs={v[4]}"
    else:
        # 变体写法："AND" 或 "k-of-N:2"；k 不写时取 ⌈size/2⌉
        variants = ([(x.split(":")[0], (int(x.split(":")[1]) if ":" in x else None))
                     for x in args.variants.split(",")]
                    if args.variants else [("AND", None), ("k-of-N", None), ("OR", None)])
        rule_of = lambda v, q: (lambda net: net.hyperedge_rule())
        net_factory_of = lambda v: make_L3_factory(v[0], v[1], args.edges, args.size, args.seed)
        measure = L3.functional_measure
        label = lambda v: (f"{v[0]}(k={v[1]})" if v[0] == "k-of-N" else v[0])

    report = {}
    table = []
    for q in (0.2, 1.0):
        for v in variants:
            t0 = time.time()
            S, F = curve_for(lambda net, v=v, q=q: rule_of(v, q)(net),
                             G_A, G_B, fractions, q, args.trials, args.seed,
                             net_factory=net_factory_of(v), measure=measure)
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
            if F is not None:
                row["F_f02"] = round(float(np.interp(0.2, fractions, F)), 4)
                row["F_f06"] = round(float(np.interp(0.6, fractions, F)), 4)
            table.append(row)
            report[f"q={q}|{label(v)}"] = list(map(float, S))
            if F is not None:
                report[f"F|q={q}|{label(v)}"] = list(map(float, F))
            print(f"  q={q} {label(v):<14} → {order:<3} 最大单步={jump:.3f} "
                  f"max_slope={row['max_slope']:.3f} p_c≈{row['pc_drop']:.2f} "
                  f"S(0.2)={row['S_f02']:.3f} S(0.6)={row['S_f06']:.3f}"
                  + (f" F(0.2)={row['F_f02']:.3f}" if F is not None else "")
                  + f" ({row['sec']}s)")

    print(f"\n=== {args.level} 相变阶判定（ER ⟨k⟩≈4, N={args.N}）===")
    print(f"{'q':>4} {'变体':>14} {'判阶':>4} {'最大单步':>8} {'max_slope':>10} {'p_c(跌)':>8}")
    for r in table:
        print(f"{r['q']:>4} {r['variant']:>14} {r['order']:>4} {r['max_jump']:>8.3f} "
              f"{r['max_slope']:>10.3f} {r['pc_drop']:>8.2f}")
    print("判读：max_slope 是网格无关的'崩得多猛'；q=1.0 应显著大于 q=0.2，才说明强耦合更脆。")

    os.makedirs(RESULTS, exist_ok=True)
    s_curves = {k: v for k, v in report.items() if not k.startswith("F|")}
    f_curves = {k.replace("F|", ""): v for k, v in report.items() if k.startswith("F|")}
    n_panels = 2 if f_curves else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 6), squeeze=False)
    axes = axes[0]
    for key, S in s_curves.items():
        q = float(key.split("|")[0].split("=")[1])
        axes[0].plot(fractions, S, "o-", markersize=3, alpha=0.9 if q == 1.0 else 0.5,
                     linestyle="-" if q == 1.0 else "--", label=key)
    axes[0].set_xlabel("初始移除比例 f")
    axes[0].set_ylabel("存活序参量 S（GCC 占比）")
    axes[0].set_title(f"{args.level} 相变阶（ER ⟨k⟩≈4, N={args.N}, {args.trials} 次重复）")
    if f_curves:
        for key, F in f_curves.items():
            q = float(key.split("|")[0].split("=")[1])
            axes[1].plot(fractions, F, "s-", markersize=3, alpha=0.9 if q == 1.0 else 0.5,
                         linestyle="-" if q == 1.0 else "--", label=key)
        axes[1].set_xlabel("初始移除比例 f")
        axes[1].set_ylabel("功能度量 F（活跃功能单元占比）")
        axes[1].set_title("功能视角：超边逻辑语义")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
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
