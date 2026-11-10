# Chooses STG's settings for each split from the tuning runs (main.py --mode select).
# Only VALIDATION plants are used here - the test plants are never looked at when choosing.
#
#   best   = the setting whose FINAL model has the highest validation r
#   sparse = the same, but only among settings that end with at most --sparse_max open gates
#            (close to the 354 markers VBS-ML kept, so the comparison with the paper is at a similar marker count)
#
# Writes <results>/chosen_config.json (read by the analysis scripts), chosen_config.txt (read by run_all.sh to
# launch the final runs) and sim_config.txt (the setting used for the simulation).
import argparse
from pathlib import Path

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--sparse_max", type=int, default=400)
args = parser.parse_args()
R = Path(args.results)

runs = ru.load_runs(R, "select")
chosen = {}
lines = []


def setting(run):
    c = run["config"]
    return {
        "stg_lambda": c["stg_lambda"], "l1_lambda": c["l1_lambda"], "epochs": c["epochs"],
        "valid_r": run["final_valid"]["r"], "valid_re": run["final_valid"]["re"],
        "gates": run["final_open_gates"], "run": run["run"],
    }


for s in ru.SPLITS:
    rows = [r for r in runs if Path(r["config"]["filename"]).name == f"data_split{s}.pt"]
    if not rows:
        print(f"split {s}: no tuning runs found - skipped")
        continue

    print(f"\nsplit {s} tuning runs (validation plants only), best first:")
    print(f"  {'stg_lambda':>10} {'l1':>7} {'epochs':>6} | {'final r':>7} {'RE%':>6} {'gates':>6} | best checkpoint r (epoch)")
    for r in sorted(rows, key=lambda r: -r["final_valid"]["r"]):
        c, f, b = r["config"], r["final_valid"], r["best_valid"]
        print(f"  {c['stg_lambda']:>10g} {c['l1_lambda']:>7g} {c['epochs']:>6} | {f['r']:7.3f} {f['re']:6.2f} {r['final_open_gates']:>6} | {b['r']:.3f} ({b['epoch']})")

    best = max(rows, key=lambda r: r["final_valid"]["r"])
    sparse_rows = [r for r in rows if 0 < r["final_open_gates"] <= args.sparse_max]
    sparse = max(sparse_rows, key=lambda r: r["final_valid"]["r"]) if sparse_rows else None
    chosen[s] = {"best": setting(best), "sparse": setting(sparse) if sparse else None}

    if best["best_valid"]["r"] - best["final_valid"]["r"] > 0.03:
        print(f"  NOTE: the chosen run peaked at epoch {best['best_valid']['epoch']} (r={best['best_valid']['r']:.3f}) "
              f"and ended lower - the model overfits late in training")
    if sparse is None:
        print(f"  no setting ended with 1-{args.sparse_max} open gates, so there is no 'sparse' setting for this split")

    # one line per distinct setting; if best and sparse are the same setting it is only trained once
    if sparse is None:
        todo = [("best", chosen[s]["best"])]
    elif ru.same_setting(chosen[s]["best"], chosen[s]["sparse"]):
        todo = [("best+sparse", chosen[s]["best"])]
    else:
        todo = [("best", chosen[s]["best"]), ("sparse", chosen[s]["sparse"])]
    for label, st in todo:
        lines.append(f"{s} {label} {st['stg_lambda']!r} {st['l1_lambda']!r} {st['epochs']}")
        print(f"  -> {label}: stg_lambda={st['stg_lambda']:g} l1={st['l1_lambda']:g} epochs={st['epochs']} "
              f"(valid r={st['valid_r']:.3f}, RE={st['valid_re']:.2f}%, {st['gates']} open gates)")

ru.write_json(R / "chosen_config.json", chosen)
(R / "chosen_config.txt").write_text("".join(line + "\n" for line in lines))

# the simulation uses split 01's sparse setting (selection is the point there), or its best one if there is no sparse one
if "01" in chosen:
    st = chosen["01"]["sparse"] or chosen["01"]["best"]
    (R / "sim_config.txt").write_text(f"{st['stg_lambda']!r} {st['l1_lambda']!r} {st['epochs']}\n")
print(f"\nwrote {R / 'chosen_config.json'}")
