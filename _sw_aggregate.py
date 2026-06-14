# -*- coding: utf-8 -*-
"""Aggregate clean 3-seed shockwave eval (outputs/logs/sw_eval/).
Per train-seed: pool success episodes across 3 eval-seeds -> mean metric.
Then mean±std over the 3 seeds. Also abs std = CV*mean_speed (un-confounded magnitude)."""
import csv, glob, statistics as st
from pathlib import Path

SW = Path("outputs/logs/sw_eval")

def read(fp):
    try:
        return list(csv.DictReader(open(fp, encoding="utf-8")))
    except FileNotFoundError:
        return []

def succ_rows(rows):
    return [r for r in rows if r.get("outcome") == "success"]

def fnum(r, c):
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else None

def per_seed_mean(rows, col):
    vs = [fnum(r, col) for r in rows]
    vs = [v for v in vs if v is not None]
    return (sum(vs) / len(vs)) if vs else None

METRICS = ["shockwave_index", "mainline_mean_speed", "throughput"]

def policy_block(label, seed_files):
    """seed_files: list of (seed_label, [csv paths to pool]). Returns dict metric-> (mean,std,per)."""
    per_metric = {m: [] for m in METRICS}
    abs_std_per = []
    n_succ_total = 0
    n_total = 0
    for sd, paths in seed_files:
        rows = []
        for p in paths:
            rows += read(p)
        n_total += len(rows)
        sr = succ_rows(rows)
        n_succ_total += len(sr)
        if not sr:
            continue
        for m in METRICS:
            v = per_seed_mean(sr, m)
            if v is not None:
                per_metric[m].append(v)
        # absolute std of mainline speed = CV * mean_speed, per episode then mean
        ab = [fnum(r, "shockwave_index") * fnum(r, "mainline_mean_speed")
              for r in sr if fnum(r, "shockwave_index") is not None and fnum(r, "mainline_mean_speed") is not None]
        if ab:
            abs_std_per.append(sum(ab) / len(ab))
    out = {}
    for m in METRICS:
        xs = per_metric[m]
        out[m] = (st.mean(xs), st.pstdev(xs), [round(x, 3) for x in xs]) if xs else (None, None, [])
    out["abs_std"] = (st.mean(abs_std_per), st.pstdev(abs_std_per), [round(x, 3) for x in abs_std_per]) if abs_std_per else (None, None, [])
    out["n_succ"] = n_succ_total
    out["n_total"] = n_total
    return label, out

def main():
    blocks = []
    # PPO / A2C: 3 train-seeds (n1,n2,n3), each pools its 3 eval-seed CSVs
    for algo, lab in (("ppo", "MAPPO (argmax)"), ("a2c", "MAA2C (stoch)")):
        sf = [(f"n{n}", [str(SW / f"{algo}_n{n}_seed{s}.csv") for s in (0, 1, 2)]) for n in (1, 2, 3)]
        blocks.append(policy_block(lab, sf))
    # Greedy: 3 eval-seeds as the 3 "seeds"
    sf = [(f"seed{s}", [str(SW / f"greedy_seed{s}.csv")]) for s in (0, 1, 2)]
    blocks.append(policy_block("Greedy (luật)", sf))

    print("=" * 92)
    print("CLEAN 3-SEED SHOCKWAVE — high-84 e2e, shield ON (PPO=argmax, A2C=stoch)")
    print("=" * 92)
    hdr = f"{'policy':16s} {'n_succ/n':>9s} {'CV(±)':>14s} {'absStd(±)':>14s} {'speed(±)':>14s} {'thru(±)':>14s}"
    print(hdr)
    def fmt(t):
        m, s, per = t
        return f"{m:.3f}±{s:.3f}" if m is not None else "  --"
    for label, o in blocks:
        print(f"{label:16s} {str(o['n_succ'])+'/'+str(o['n_total']):>9s} "
              f"{fmt(o['shockwave_index']):>14s} {fmt(o['abs_std']):>14s} "
              f"{fmt(o['mainline_mean_speed']):>14s} {fmt(o['throughput']):>14s}")
    print()
    for label, o in blocks:
        print(f"  {label}: CV per-seed={o['shockwave_index'][2]}  absStd per-seed={o['abs_std'][2]}")

if __name__ == "__main__":
    main()
