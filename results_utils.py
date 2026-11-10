# Shared helpers for the analysis scripts (select_config.py, baselines.py, stability.py, score_sim.py, summarise.py).
import json
import math
from pathlib import Path

import numpy as np
import torch

SPLITS = ["01", "02", "03", "04"]
ROLES = ["best", "sparse"]  # the two STG settings chosen per split (see select_config.py)
LD_R = 0.8  # two markers count as "the same signal" when |r| between them is at least this

# 2014 results from Yan et al. (2023) Plant Methods 19:96, splits 1-4.
# Table 1 = mean relative error (%), Table 2 = Pearson r between predicted and observed adjusted yield.
PAPER = {
    "VBS-ML (354)": {"re": [6.49, 7.08, 7.58, 7.04], "r": [0.66, 0.50, 0.47, 0.52]},
    "LMM": {"re": [7.38, 7.21, 7.44, 7.31], "r": [0.52, 0.49, 0.50, 0.47]},
    "BayesA": {"re": [7.28, 7.16, 7.37, 7.25], "r": [0.54, 0.50, 0.51, 0.48]},
    "BayesB": {"re": [7.20, 7.09, 7.36, 7.21], "r": [0.55, 0.51, 0.51, 0.49]},
    "Naive-ML": {"re": [7.61, 7.56, 8.31, 7.61], "r": [0.54, 0.41, 0.39, 0.42]},
}


def load(path):
    """torch.load for our own data files (they contain lists of marker names, so weights_only must be off)."""
    return torch.load(path, weights_only=False)


def valid_is_test(d):
    """True when the file has no separate validation plants (make_splits.py --val_frac 0).
    Older files used the key 'is_valid_set' for the same thing."""
    return bool(d.get("valid_is_test", d.get("is_valid_set", False)))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    """Write to a temporary file first, so a crash never leaves a half-written results file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load_runs(results, mode=None):
    """Every finished training run (one JSON per run, written by main.py)."""
    runs = []
    for f in sorted(Path(results, "runs").glob("*.json")):
        r = read_json(f)
        if mode is None or r["config"]["mode"] == mode:
            runs.append(r)
    return runs


def same_setting(cfg, setting):
    return (
        math.isclose(cfg["stg_lambda"], setting["stg_lambda"], rel_tol=1e-9)
        and math.isclose(cfg["l1_lambda"], setting["l1_lambda"], rel_tol=1e-9)
        and int(cfg["epochs"]) == int(setting["epochs"])
    )


def report_roles(chosen):
    """'sparse' is only reported on its own when it differs from 'best' on at least one split."""
    differs = any(
        chosen[s].get("sparse") and not same_setting(chosen[s]["best"], chosen[s]["sparse"]) for s in chosen
    )
    return ROLES if differs else ["best"]


def final_runs(results, split, setting):
    """The final (fixed-epoch, full 90% training set) runs for one split and one chosen setting, sorted by seed."""
    if not setting:
        return []
    name = f"data_split{split}_full.pt"
    runs = [
        r for r in load_runs(results, "fixed")
        if Path(r["config"]["filename"]).name == name and same_setting(r["config"], setting)
    ]
    return sorted(runs, key=lambda r: r["config"]["seed"])


def load_mu(run):
    return torch.load(run["mu_path"]).float().numpy()


def standardise(X):
    """Columns scaled to mean 0 / sd 1, so (Z[:, a].T @ Z[:, b]) / (n - 1) is the correlation between markers.
    Markers with no variation get a column of zeros (correlation 0 with everything)."""
    X = X.float()
    sd = X.std(0)
    Z = X - X.mean(0)
    Z /= torch.where(sd > 0, sd, torch.ones_like(sd))
    Z[:, sd == 0] = 0.0
    return Z


def abs_corr(Z, a, b):
    """|correlation| between marker sets a and b (a len(a) x len(b) matrix). A marker always matches itself."""
    a = torch.as_tensor(np.asarray(a), dtype=torch.long, device=Z.device)
    b = torch.as_tensor(np.asarray(b), dtype=torch.long, device=Z.device)
    C = (Z[:, a].T @ Z[:, b]).abs() / (Z.shape[0] - 1)
    C[a[:, None] == b[None, :]] = 1.0
    return C


def mean_sd(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
