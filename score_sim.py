# Scores the simulation runs: did STG find the markers that really affect the (simulated) yield?
#
#   recall     share of the true markers that were selected
#   precision  share of the selected markers that are true markers
#   LD-aware   a hit also counts when the selected marker and the true marker have |r| >= 0.8
#
# Scored for STG's own selection (every open gate), for STG's top-m markers (m = number of true markers),
# and for the cheapest selector: the top-m markers by correlation with yield on the training plants.
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
args = parser.parse_args()
R = Path(args.results)
device = "cuda" if torch.cuda.is_available() else "cpu"

runs = [r for r in ru.load_runs(R, "fixed") if Path(r["config"]["filename"]).name.startswith("sim_")]
if not runs:
    raise SystemExit("no simulation runs found")

bases = {}


def base_data(path):
    """Standardised markers of all plants (for marker-marker correlations) and the centred training markers
    (for the correlation-with-yield baseline) of the base split. Loaded once per base file."""
    if path not in bases:
        d = ru.load(path)
        parts = [d["X_train"], d["X_test"]] if ru.valid_is_test(d) else [d["X_train"], d["X_valid"], d["X_test"]]
        Xtr = d["X_train"].float().numpy()
        bases[path] = (ru.standardise(torch.cat(parts).to(device)), Xtr - Xtr.mean(0))
    return bases[path]


def scores(S, T, Z):
    S, T = np.asarray(S), np.asarray(T)
    hit = np.isin(S, T)
    out = {"n_selected": int(len(S)), "recall": float(np.isin(T, S).mean()),
           "precision": float(hit.mean()) if len(S) else None}
    if len(S) == 0:
        out.update({"recall_ld": 0.0, "precision_ld": None})
        return out
    C = ru.abs_corr(Z, T, S) #true markers x selected markers
    out["recall_ld"] = (C.max(1).values >= ru.LD_R).float().mean().item()
    out["precision_ld"] = (C.max(0).values >= ru.LD_R).float().mean().item()
    return out


per_run = []
for run in runs:
    sim = ru.load(run["config"]["filename"])
    T = sim["true_idx"].numpy()
    m = len(T)
    Z, Xc = base_data(sim["base"])
    mu = ru.load_mu(run)

    ytr = sim["Y_train"].numpy()
    num = Xc.T @ (ytr - ytr.mean())
    den = np.sqrt((Xc ** 2).sum(0)) * np.sqrt(((ytr - ytr.mean()) ** 2).sum())
    marg = np.argsort(-np.abs(np.divide(num, den, out=np.zeros_like(num), where=den > 0)))

    per_run.append({
        "condition": sim["condition"], "n_true": m, "rep": sim["rep"], "run": run["run"],
        "test_r": run["test"]["r"],
        "stg_open": scores(np.where(mu > 0)[0], T, Z),
        "stg_top_m": scores(np.argsort(-mu)[:m], T, Z),
        "marginal_top_m": scores(marg[:m], T, Z),
    })
    print(f"{run['run']}: open gates {per_run[-1]['stg_open']['n_selected']}, "
          f"top-m recall {per_run[-1]['stg_top_m']['recall']:.2f} (LD {per_run[-1]['stg_top_m']['recall_ld']:.2f})")

groups = defaultdict(list)
for r in per_run:
    groups[(r["condition"], r["n_true"])].append(r)

summary = []
for (cond, m), rs in sorted(groups.items()):
    row = {"condition": cond, "n_true": m, "reps": len(rs), "test_r": ru.mean_sd([r["test_r"] for r in rs])[0]}
    for sel in ("stg_open", "stg_top_m", "marginal_top_m"):
        row[sel] = {k: ru.mean_sd([r[sel][k] for r in rs])[0]
                    for k in ("n_selected", "recall", "recall_ld", "precision", "precision_ld")}
    summary.append(row)

ru.write_json(R / "sim_scores.json", {"summary": summary, "runs": per_run})
print(f"wrote {R / 'sim_scores.json'}")
