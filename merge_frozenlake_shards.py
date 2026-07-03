#!/usr/bin/env python
"""Merge N FrozenLake eval shards into one combined output with aggregate
metrics (per-level, in-distribution, overall). Adds EM to the
metrics.json."""

import argparse
import json
import os
import sys
from collections import Counter, deque

_DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def bfs_dist(layout, src, tgt, L):
    """BFS distance from src to tgt over non-hole cells. None if unreachable."""
    if src == tgt:
        return 0
    visited = {src}
    q = deque([(src, 0)])
    while q:
        p, d = q.popleft()
        r, c = divmod(p, L)
        for dr, dc in _DIR.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < L and 0 <= nc < L):
                continue
            np = nr * L + nc
            if np in visited or layout[nr][nc] == "H":
                continue
            if np == tgt:
                return d + 1
            visited.add(np)
            q.append((np, d + 1))
    return None


def is_em(actions, layout, start, tgt, L, n_opt):
    """Simulator-realized trajectory coincides with some optimal trajectory:
    reaches goal AND consumes exactly n_opt actions before sim stop."""
    if actions is None or n_opt is None:
        return False
    r, c = divmod(start, L)
    consumed = 0
    reached = False
    for i, act in enumerate(actions):
        consumed = i + 1
        if act not in _DIR:
            return False
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            continue  # wall hit: no-op
        r, c = nr, nc
        if layout[r][c] == "H":
            return False
        if layout[r][c] == "G":
            reached = True
            break
    return reached and consumed == n_opt


def progress_rate(actions, layout, start, tgt, L, n_opt):
    """(longest strictly-distance-decreasing legal prefix) / n_opt."""
    if actions is None or n_opt is None:
        return 0.0
    if n_opt == 0:
        return 1.0
    cur = start
    cur_d = bfs_dist(layout, cur, tgt, L)
    if cur_d is None:
        return 0.0
    matched = 0
    for act in actions:
        if act not in _DIR:
            break
        r, c = divmod(cur, L)
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            break
        if layout[nr][nc] == "H":
            break
        nb = nr * L + nc
        nd = bfs_dist(layout, nb, tgt, L)
        if nd is None or nd >= cur_d:
            break
        matched += 1
        cur, cur_d = nb, nd
    return matched / n_opt


def aggregate(group):
    n = len(group)
    if not n:
        return None
    strict = sum(1 for r in group if r["strict_correct"])
    turn1 = sum(1 for r in group if r["turn1_strict_correct"])
    return {
        "n": n,
        "EM": sum(1 for r in group if r["EM"]) / n,
        "PR": sum(r["PR"] for r in group) / n,
        "strict_success_rate": strict / n,
        "lenient_success_rate": sum(1 for r in group if r["lenient_correct"]) / n,
        "optimal_rate": sum(1 for r in group if r["optimal"]) / n,
        "turn1_strict_rate": turn1 / n,
        "final_strict_rate": strict / n,
        "turn1_to_final_delta": (strict - turn1) / n,
        "avg_turns": sum(r["n_turns"] for r in group) / n,
        "avg_route_calls": sum(r["n_route_calls"] for r in group) / n,
        "format_error_rate": sum(1 for r in group if r["final_kind"] == "format_error") / n,
        "max_turns_rate": sum(1 for r in group if r["final_kind"] == "max_turns") / n,
        "clean_terminate_rate": sum(1 for r in group if r["final_kind"] == "terminate") / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", required=True,
                    help="comma-separated shard output dirs")
    ap.add_argument("--output_dir", required=True, help="merged output dir")
    ap.add_argument("--eval_dataset",
                    default="data/FrozenLake/eval_data/eval_data.json")
    args = ap.parse_args()

    shard_dirs = [d.strip() for d in args.shard_dirs.split(",") if d.strip()]
    eval_full = json.load(open(args.eval_dataset))
    md = {ex["id"]: ex["metadata"] for ex in eval_full}

    all_recs = []
    for sd in shard_dirs:
        p = os.path.join(sd, "outputs.json")
        if not os.path.exists(p):
            sys.exit(f"Missing {p}")
        recs = json.load(open(p))
        print(f"  loaded {len(recs)} from {sd}")
        all_recs.extend(recs)
    print(f"Total records: {len(all_recs)}")
    print(f"Level dist: {dict(Counter(r['level'] for r in all_recs))}")

    for r in all_recs:
        m = md[r["id"]]
        actions = r["final_actions"]
        r["EM"] = is_em(actions, m["layout"], m["start_pos"], m["target_pos"],
                        m["level"], r["optimal_path_len"])
        r["PR"] = progress_rate(actions, m["layout"], m["start_pos"], m["target_pos"],
                                m["level"], r["optimal_path_len"])

    by_level = {}
    for lvl in sorted({r["level"] for r in all_recs}):
        by_level[str(lvl)] = aggregate([r for r in all_recs if r["level"] == lvl])
    in_dist = aggregate([r for r in all_recs if r["level"] in (3, 4)])
    overall = aggregate(all_recs)
    metrics = {"by_level": by_level, "in_distribution_3_4": in_dist, "overall": overall}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "outputs.json")
    with open(out_path, "w") as f:
        json.dump(all_recs, f, default=str)
    print(f"Outputs: {out_path}")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")

    print("\n=== Merged FrozenLake Eval Metrics ===")
    hdr = f"{'Group':<22} {'n':>4} {'EM':>7} {'PR':>7} {'Lenient':>9} {'Strict':>8} {'Optimal':>9} {'avg_t':>6}"
    print(hdr)
    print("-" * len(hdr))
    for lvl in sorted(metrics["by_level"]):
        g = metrics["by_level"][lvl]
        print(f"{f'{lvl}x{lvl}':<22} {g['n']:>4} "
              f"{100*g['EM']:>6.2f}% {100*g['PR']:>6.2f}% "
              f"{100*g['lenient_success_rate']:>8.2f}% {100*g['strict_success_rate']:>7.2f}% "
              f"{100*g['optimal_rate']:>8.2f}% {g['avg_turns']:>6.2f}")
    for label, g in [("in-distribution (3+4)", in_dist), ("overall", overall)]:
        if g is None:
            continue
        print(f"{label:<22} {g['n']:>4} "
              f"{100*g['EM']:>6.2f}% {100*g['PR']:>6.2f}% "
              f"{100*g['lenient_success_rate']:>8.2f}% {100*g['strict_success_rate']:>7.2f}% "
              f"{100*g['optimal_rate']:>8.2f}% {g['avg_turns']:>6.2f}")


if __name__ == "__main__":
    main()
