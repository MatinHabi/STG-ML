#THIS SCRIPT ANSWERS THE QUESTION "how good are STG's selected markers?"
#Idea: take a set of K markers, fit a simple linear model (ridge regression) on the training plants using ONLY those
#markers, and score its predictions on the test plants. Good markers -> high r / low RE even when K is small.
#The same ridge model is used for every marker set, so the only thing being compared is WHICH markers were picked.
#
#STG's top-K markers (ranked by their learnt mu) are compared at the same K against:
#   - random markers, many draws          -> the floor: what you get with no selection at all
#   - the K markers most correlated with yield on the training plants -> the cheapest possible selector
#   - LASSO, tuned to keep about K markers -> the classic sparse linear selector
#   - all 17,114 markers                  -> the reference (ridge on all markers is close to the paper's LMM / GBLUP)
#
#   python baselines.py --check     ridge on all markers per split, compared with the paper's LMM row (checks the data matches)
#   python baselines.py --markers   the refit test, using the mu files from the final STG runs
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV, lasso_path #Ridge regression: a linear model with a penalty alpha that keeps the weights small so it doesnt overfit with many markers

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--check", action="store_true")
parser.add_argument("--markers", action="store_true")
parser.add_argument("--quick", action="store_true") #tiny version for the smoke test
parser.add_argument("--no_lasso", action="store_true")
args = parser.parse_args()
R = Path(args.results)

KS = [25, 100] if args.quick else [25, 50, 100, 250, 354, 500, 1000, 2000]
N_RANDOM = 3 if args.quick else 30
ALPHAS = np.logspace(-1, 7, 17) #17 alpha values spread evenly on a log scale from 10^-1 to 10^7; RidgeCV keeps the best one
                                #(chosen by leave-one-out cross-validation on the TRAINING plants only)
NATURAL_MAX = 5000 #STG's own selection (mu > 0) is only refitted when it has at most this many markers


def load_split(s):
    d = ru.load(f"data_split{s}_full.pt")
    Xtr, ytr = d["X_train"].numpy(), d["Y_train"].numpy()
    if not ru.valid_is_test(d): #separate validation plants: give them back to training, ridge doesn't need them
        Xtr = np.vstack([Xtr, d["X_valid"].numpy()]) #putting validation plants rows under training plants
        ytr = np.concatenate([ytr, d["Y_valid"].numpy()])
    return Xtr, ytr, d["X_test"].numpy(), d["Y_test"].numpy(), float(d["mean"]), float(d["std"])


def score(data, cols=None):
    Xtr, ytr, Xte, yte, mean, std = data
    A, B = (Xtr, Xte) if cols is None else (Xtr[:, cols], Xte[:, cols]) #cols=None means use all 17,114 markers
    ridge = RidgeCV(alphas=ALPHAS).fit(A, ytr) #learns one weight per marker in cols, keeps the best alpha
    pred = ridge.predict(B) #predictions for the TEST plants
    y_kg, pred_kg = yte * std + mean, pred * std + mean #back to kg/ha for the paper's relative error
    return {
        "r": float(np.nan_to_num(np.corrcoef(pred, yte)[0, 1])),
        "re": float(np.mean(np.abs(y_kg - pred_kg) / np.abs(y_kg)) * 100),
        "alpha": float(ridge.alpha_),
        "n_markers": int(Xtr.shape[1] if cols is None else len(cols)),
    }


def marginal_rank(Xtr, ytr):
    """Markers ordered by |correlation with yield| on the training plants (strongest first)."""
    Xc = Xtr - Xtr.mean(0)
    yc = ytr - ytr.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum())
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.argsort(-np.abs(r))


def lasso_sets(Xtr, ytr, ks):
    """For each K: the markers LASSO keeps at the largest penalty that keeps at least K markers (the top K by |weight|).
    Walks down the penalty path one value at a time and stops once the largest K is reached."""
    sd = Xtr.std(0)
    keep = np.where(sd > 0)[0]
    Xs = ((Xtr[:, keep] - Xtr[:, keep].mean(0)) / sd[keep]).astype(np.float32) #standardised, so the penalty treats every marker the same
    yc = (ytr - ytr.mean()).astype(np.float32)
    a_max = np.max(np.abs(Xs.T @ yc)) / len(yc) #smallest penalty at which LASSO keeps no markers at all
    sets, coef = {}, None
    todo = sorted(k for k in ks if k < len(keep))
    for a in a_max * np.logspace(-0.05, -3, 40):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, coefs, _ = lasso_path(Xs, yc, alphas=[a], coef_init=coef, max_iter=200, tol=1e-3)
        coef = coefs[:, 0]
        nnz = int((coef != 0).sum())
        while todo and nnz >= todo[0]:
            k = todo.pop(0)
            sets[k] = keep[np.argsort(-np.abs(coef))[:k]]
        if not todo:
            break
    return sets


def check():
    splits = ["01"] if args.quick else ru.SPLITS
    out = {}
    print("ridge on ALL markers (train on 90%, score the paper's test plants) vs the paper's LMM:")
    for i, s in enumerate(ru.SPLITS):
        if s not in splits:
            continue
        t0 = time.time()
        out[s] = score(load_split(s))
        lmm_r, lmm_re = ru.PAPER["LMM"]["r"][i], ru.PAPER["LMM"]["re"][i]
        print(f"  split {s}: r = {out[s]['r']:.3f} (paper LMM {lmm_r:.2f}) || RE = {out[s]['re']:.2f}% (paper LMM {lmm_re:.2f}%)"
              f" || alpha {out[s]['alpha']:g} || {time.time() - t0:.0f}s")
        if out[s]["alpha"] in (ALPHAS[0], ALPHAS[-1]):
            print("    note: best alpha is at the edge of the range searched")
    ru.write_json(R / "data_check.json", out)
    print("if r is within ~0.02-0.03 of the paper's LMM on every split, your data matches theirs")


def markers():
    chosen = ru.read_json(R / "chosen_config.json")
    check_path = R / "data_check.json"
    checked = ru.read_json(check_path) if check_path.exists() else {}
    out = {"ks": KS, "splits": {}}
    sets_out = {}

    for s in ru.SPLITS:
        if s not in chosen:
            continue
        t0 = time.time()
        data = load_split(s)
        Xtr, ytr = data[0], data[1]
        p = Xtr.shape[1]
        ks = [k for k in KS if k < p]
        rng = np.random.default_rng(int(s))
        res = {"all": checked.get(s) or (None if args.quick else score(data)),
               "random": {}, "marginal": {}, "lasso": {}, "stg": {}}

        order = marginal_rank(Xtr, ytr)
        lasso = {} if (args.quick or args.no_lasso) else lasso_sets(Xtr, ytr, ks)
        sets_out[s] = {"marginal": {str(k): order[:k].tolist() for k in ks},
                       "lasso": {str(k): (lasso[k].tolist() if k in lasso else None) for k in ks}}

        for k in ks:
            res["random"][str(k)] = [score(data, rng.choice(p, k, replace=False)) for _ in range(N_RANDOM)]
            res["marginal"][str(k)] = score(data, order[:k])
            res["lasso"][str(k)] = score(data, lasso[k]) if k in lasso else None

        for role in ru.ROLES:
            st = chosen[s].get(role)
            if role == "sparse" and st and "best" in res["stg"] and ru.same_setting(st, chosen[s]["best"]):
                res["stg"]["sparse"] = res["stg"]["best"] #same setting = same runs, no need to refit twice
                continue
            runs = ru.final_runs(R, s, st)
            if not runs:
                continue
            per_k = {str(k): [] for k in ks}
            natural = []
            for run in runs:
                mu = ru.load_mu(run)
                ranked = np.argsort(-mu) #markers from highest gate value to lowest
                for k in ks:
                    per_k[str(k)].append(score(data, ranked[:k]))
                picked = np.where(mu > 0)[0] #STG's own selection: every open gate
                if 0 < len(picked) <= NATURAL_MAX:
                    rand = [score(data, rng.choice(p, len(picked), replace=False)) for _ in range(5)]
                    natural.append({**score(data, picked), "seed": run["config"]["seed"],
                                    "random_r": float(np.mean([x["r"] for x in rand])),
                                    "random_re": float(np.mean([x["re"] for x in rand]))})
                else:
                    natural.append({"seed": run["config"]["seed"], "n_markers": int(len(picked)), "r": None, "re": None})
            res["stg"][role] = {"setting": chosen[s][role], "per_k": per_k, "natural": natural}

        out["splits"][s] = res
        ru.write_json(R / "refit.json", out) #saved after every split, so a crash keeps the finished ones
        print(f"split {s}: refit test done ({time.time() - t0:.0f}s)")

    ru.write_json(R / "baseline_sets.json", sets_out) #used by stability.py for the redundancy comparison


if __name__ == "__main__":
    if not (args.check or args.markers):
        parser.error("use --check and/or --markers")
    if args.check:
        check()
    if args.markers:
        markers()
