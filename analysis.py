"""Baseline (default gamma=1, omega=1) vs. gamma/omega chosen on a validation window,
for multilayer (Mucha-style) Louvain community detection on the CollegeMsg messaging network.
Main metric: intra-community fraction of test-window messages (higher is better)."""
import gzip, io, json, os, sys, time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import adjusted_rand_score

URL = "https://snap.stanford.edu/data/CollegeMsg.txt.gz"
SEEDS = [42, 43, 44]     # seed 42 is the primary seed; 43/44 used for stability runs
MAX_ROWS = 200_000       # subsample cap (CollegeMsg has ~60k rows, so this is a no-op)
K_SLICES = 6             # equal-width time slices per window
LAMBDA = 0.5             # weight of the partition-instability penalty in selection
GRID = [(g, w) for g in (0.5, 1.0, 2.0) for w in (0.25, 1.0, 2.0)]  # contains (1, 1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, retries=5):
    """GET with timeout=60; retry with exponential backoff on HTTP 429/5xx and network errors."""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                log(f"HTTP {r.status_code}; retrying in {2 ** i}s")
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            log(f"request error {e}; retrying in {2 ** i}s")
            time.sleep(2 ** i)
    raise RuntimeError("download failed after retries")


def load():
    """Load time-ordered messages (src, dst, ts); drop self-loops; cap at MAX_ROWS."""
    raw = gzip.decompress(download(URL))
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, names=["src", "dst", "ts"])
    df = df[df.src != df.dst].sort_values("ts", kind="stable").reset_index(drop=True)
    return df.iloc[:MAX_ROWS]


def split(df):
    """Time-ordered split: first 60% train, next 20% validation, last 20% test (no shuffling)."""
    n = len(df)
    a, b = int(0.6 * n), int(0.8 * n)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def partition(train, gamma, omega, seed):
    """Multilayer Louvain: one node per (user, time slice); intra-slice edges weighted by
    message counts; inter-slice edges (same user, consecutive slices) weighted by omega.
    Returns user -> community label taken from the latest training slice."""
    slice_id = np.asarray(pd.cut(train.ts, K_SLICES, labels=False), dtype=int)
    users = [int(u) for u in np.union1d(train.src.values, train.dst.values)]
    G = nx.Graph()
    G.add_nodes_from((u, k) for u in users for k in range(K_SLICES))
    cnt = Counter()
    for u, v, k in zip(train.src.tolist(), train.dst.tolist(), slice_id.tolist()):
        cnt[(min(u, v), max(u, v), k)] += 1
    for (u, v, k), c in cnt.items():
        G.add_edge((u, k), (v, k), weight=float(c))
    for u in users:
        for k in range(K_SLICES - 1):
            G.add_edge((u, k), (u, k + 1), weight=float(omega))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=gamma, seed=seed)
    lab = {node: i for i, c in enumerate(comms) for node in c}
    return {u: lab[(u, K_SLICES - 1)] for u in users}


def evaluate(part, df):
    """Fraction of messages whose two endpoints share a community. Unseen users are singletons."""
    cu = [part.get(u, ("new", u)) for u in df.src.tolist()]
    cv = [part.get(v, ("new", v)) for v in df.dst.tolist()]
    return float(np.mean([a == b for a, b in zip(cu, cv)]))


def stability(parts):
    """Mean pairwise adjusted Rand index across seeds, over training users."""
    users = sorted(parts[0])
    aris = [adjusted_rand_score([p[u] for u in users], [q[u] for u in users])
            for i, p in enumerate(parts) for q in parts[i + 1:]]
    return float(np.mean(aris))


def final(train, test, gamma, omega):
    """Run all seeds with one (gamma, omega) setting and score on the test window."""
    parts = [partition(train, gamma, omega, s) for s in SEEDS]
    scores = [evaluate(p, test) for p in parts]
    return {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "ncomm": float(np.mean([len(set(p.values())) for p in parts])),
            "ari": stability(parts)}


def main():
    df = load()
    train, val, test = split(df)
    log(f"rows={len(df)} train={len(train)} val={len(val)} test={len(test)}")
    # Selection uses ONLY the validation window (partitions built from the training window).
    best = None
    for g, w in GRID:
        parts = [partition(train, g, w, s) for s in SEEDS]
        val_intra = float(np.mean([evaluate(p, val) for p in parts]))
        ari = stability(parts)
        obj = val_intra - LAMBDA * (1 - ari)
        log(f"gamma={g} omega={w}: val_intra={val_intra:.4f} ari={ari:.3f} objective={obj:.4f}")
        if best is None or obj > best[0]:
            best = (obj, g, w)
    _, g_sel, w_sel = best
    log(f"selected gamma={g_sel} omega={w_sel}; scoring test window once for both models")
    base = final(train, test, 1.0, 1.0)
    imp = final(train, test, g_sel, w_sel)
    for name, r in (("baseline", base), ("improved", imp)):
        log(f"{name}: test_intra={r['mean']:.4f}+-{r['std']:.4f} ncomm={r['ncomm']:.1f} ari={r['ari']:.3f}")
    if len(test) == 0 or not np.isfinite([base["mean"], imp["mean"]]).all():
        log("no meaningful metrics produced")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline (γ=1, ω=1)", "improved (validation-selected)"], [base["mean"], imp["mean"]],
            yerr=[base["std"], imp["std"]], color=["grey", "steelblue"], capsize=6)
    plt.ylabel("test-window intra-community fraction")
    plt.tight_layout()
    plt.savefig("figures/test_intra_fraction.png", dpi=120)
    plt.close()

    results = {
        "question": ("Can choosing the resolution γ and temporal coupling ω on an earlier validation "
                     "window give communities that better explain later interactions than the default "
                     "γ=1, ω=1 setting on a public temporal messaging network?"),
        "method": ("Time-ordered 60/20/20 split of CollegeMsg messages. Multilayer Louvain communities "
                   "are built on the training window; γ, ω are selected on the validation window by "
                   "intra-community fraction minus a partition-instability penalty; the test window is "
                   "scored once for both models over seeds 42-44."),
        "dataset": {
            "name": "UCI/SNAP CollegeMsg temporal messaging network",
            "source_url": URL,
            "n_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "test-window intra-community interaction fraction",
        "higher_is_better": True,
        "baseline": "Multilayer (Mucha-style) Louvain over time slices with default γ=1 and ω=1.",
        "improved": ("Same multilayer Louvain, with γ and ω chosen from a small grid by maximizing "
                     "validation-window intra-community fraction penalized by seed instability (1 - ARI)."),
        "metrics": {
            "baseline_score": base["mean"],
            "improved_score": imp["mean"],
            "improvement": imp["mean"] - base["mean"],
            "baseline_score_std": base["std"],
            "improved_score_std": imp["std"],
            "baseline_n_communities": base["ncomm"],
            "improved_n_communities": imp["ncomm"],
            "baseline_stability_ari": base["ari"],
            "improved_stability_ari": imp["ari"],
            "selected_gamma": g_sel,
            "selected_omega": w_sel,
        },
        "notes": ("Louvain is applied to the whole supra-graph with a single resolution parameter, an "
                  "approximation of Mucha's layer-wise null model; this is not the LAGO algorithm of the "
                  "motivating paper. Interactions are treated as undirected and unweighted by direction. "
                  "Communities are fixed from the training window; each user takes its label from the "
                  "latest training slice; users unseen in training are singletons. The validation window "
                  "is a time-ordered part of the non-test data, never the test set. Single dataset, "
                  "small grid, 3 seeds; differences may be small relative to seed variation."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json and figures/test_intra_fraction.png")


if __name__ == "__main__":
    main()
