# How STABLE and how REDUNDANT are STG's selected markers?
#
#   stability   do different seeds (same split) and different splits pick the same markers?
#               exact     = share of markers that the other run also picked
#               LD-aware  = share of markers that the other run picked OR replaced by a marker correlated with it at |r| >= 0.8
#               Low exact but high LD-aware overlap = STG finds the same regions of the genome but swaps between
#               neighbouring markers that carry the same signal.
#   redundancy  inside one selected set, what share of markers has a near-copy (|r| >= 0.8) also in the set?
#               Near-copies waste slots. Compared with random sets, the top-K by correlation with yield, and LASSO.
#   top markers which markers STG keeps most often across all seeds and splits (written to a CSV with their names)
#
# Marker-marker correlations are computed on all 10,375 plants (no yield involved, so the test plants can be used).
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
R = Path(args.results)

KS = [25, 100] if args.quick else [50, 100, 354, 500, 1000]
NATURAL_LD_MAX = 3000 #LD-aware overlap of STG's own selection (mu > 0) is skipped above this size (too slow)
device = "cuda" if torch.cuda.is_available() else "cpu"

d = ru.load("data_split01_full.pt")
names = d["marker_names"]
Z = ru.standardise(torch.cat([d["X_train"], d["X_test"]]).to(device)) #all plants: 90% training + 10% test
del d
p = Z.shape[1]
chosen = ru.read_json(R / "chosen_config.json")
sets_path = R / "baseline_sets.json"
baseline_sets = ru.read_json(sets_path) if sets_path.exists() else {}
rng = np.random.default_rng(0)


def overlap(a, b, ld=True):
    """Average of 'share of a found in b' and 'share of b found in a' (exact, and LD-aware)."""
    if len(a) == 0 or len(b) == 0:
        return None, None
    common = len(np.intersect1d(a, b))
    exact = (common / len(a) + common / len(b)) / 2
    if not ld:
        return exact, None
    C = ru.abs_corr(Z, a, b)
    found_a = (C.max(1).values >= ru.LD_R).float().mean().item()
    found_b = (C.max(0).values >= ru.LD_R).float().mean().item()
    return exact, (found_a + found_b) / 2


def redundancy(S):
    """Share of markers in S with a near-copy (|r| >= 0.8) elsewhere in S, and the mean |r| between pairs in S."""
    if len(S) < 2:
        return None
    C = ru.abs_corr(Z, S, S)
    C.fill_diagonal_(0.0)
    k = len(S)
    return {"near_copy": (C.max(1).values >= ru.LD_R).float().mean().item(),
            "mean_abs_r": (C.sum() / (k * (k - 1))).item()}


def summary(pairs):
    ex, ld = ru.mean_sd([x[0] for x in pairs]), ru.mean_sd([x[1] for x in pairs])
    return {"exact": ex[0], "exact_sd": ex[1], "ld": ld[0], "ld_sd": ld[1], "pairs": len(pairs)}


def mean_dicts(items):
    items = [x for x in items if x]
    if not items:
        return None
    return {key: float(np.mean([x[key] for x in items])) for key in items[0]}


out = {}
for role in ru.report_roles(chosen):
    mus = {}
    for s in ru.SPLITS:
        runs = ru.final_runs(R, s, chosen.get(s, {}).get(role))
        if runs:
            mus[s] = {run["config"]["seed"]: ru.load_mu(run) for run in runs}
    if not mus:
        continue
    print(f"[{role}] {sum(len(v) for v in mus.values())} runs over splits {sorted(mus)}")
    res = {"within": {}, "across": {}, "random": {}, "redundancy": {}, "natural": {}}

    for k in [k for k in KS if k < p]:
        top = {s: {seed: np.argsort(-mu)[:k] for seed, mu in runs.items()} for s, runs in mus.items()}
        within = [overlap(top[s][a], top[s][b]) for s in top for a, b in combinations(sorted(top[s]), 2)]
        across = [overlap(top[s1][seed], top[s2][seed])
                  for s1, s2 in combinations(sorted(top), 2) for seed in top[s1] if seed in top[s2]]
        rand = [overlap(rng.choice(p, k, replace=False), rng.choice(p, k, replace=False)) for _ in range(5)]
        res["within"][str(k)] = summary(within)
        res["across"][str(k)] = summary(across)
        res["random"][str(k)] = summary(rand)

        res["redundancy"][str(k)] = {
            "stg": mean_dicts([redundancy(S) for s in top for S in top[s].values()]),
            "random": mean_dicts([redundancy(rng.choice(p, k, replace=False)) for _ in range(5)]),
            "marginal": mean_dicts([redundancy(np.array(baseline_sets[s]["marginal"][str(k)]))
                                    for s in mus if baseline_sets.get(s, {}).get("marginal", {}).get(str(k))]),
            "lasso": mean_dicts([redundancy(np.array(baseline_sets[s]["lasso"][str(k)]))
                                 for s in mus if baseline_sets.get(s, {}).get("lasso", {}).get(str(k))]),
        }
        print(f"  K={k}: within-split exact {res['within'][str(k)]['exact']}, LD {res['within'][str(k)]['ld']}")

    # STG's own selection (every open gate, mu > 0) - its size differs from run to run
    picked = {s: {seed: np.where(mu > 0)[0] for seed, mu in runs.items()} for s, runs in mus.items()}
    sizes = [len(S) for s in picked for S in picked[s].values()]
    small = all(0 < n <= NATURAL_LD_MAX for n in sizes)
    within = [overlap(picked[s][a], picked[s][b], ld=small) for s in picked for a, b in combinations(sorted(picked[s]), 2)]
    across = [overlap(picked[s1][seed], picked[s2][seed], ld=small)
              for s1, s2 in combinations(sorted(picked), 2) for seed in picked[s1] if seed in picked[s2]]
    res["natural"] = {"size_mean": float(np.mean(sizes)), "size_min": int(min(sizes)), "size_max": int(max(sizes)),
                      "within": summary([x for x in within if x[0] is not None]),
                      "across": summary([x for x in across if x[0] is not None]),
                      "random_exact": float(np.mean(sizes)) / p,
                      "redundancy": mean_dicts([redundancy(S) for s in picked for S in picked[s].values()]) if small else None}

    # how often each marker is kept, over every run of this role
    all_mu = [mu for runs in mus.values() for mu in runs.values()]
    freq = np.mean([mu > 0 for mu in all_mu], axis=0)
    mean_mu = np.mean(all_mu, axis=0)
    ranked = np.lexsort((-mean_mu, -freq))[:50] #most often kept first, ties broken by mean mu
    with open(R / f"top_markers_{role}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "marker", "column", "kept_in_share_of_runs", "mean_mu"])
        for i, j in enumerate(ranked, 1):
            w.writerow([i, names[j], int(j), f"{freq[j]:.3f}", f"{mean_mu[j]:.4f}"])
    out[role] = res

ru.write_json(R / "stability.json", out)
print(f"wrote {R / 'stability.json'}")
