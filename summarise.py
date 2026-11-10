# Collects every result into one readable report: <results>/summary.md
import argparse
from pathlib import Path

import numpy as np

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
args = parser.parse_args()
R = Path(args.results)


def get(name):
    path = R / name
    return ru.read_json(path) if path.exists() else None


def num(x, f=".3f"):
    return "–" if x is None else format(x, f)


def pm(values, f=".3f"):
    mean, sd = ru.mean_sd(values)
    return "–" if mean is None else f"{mean:{f}} ± {sd:{f}}"


lines = ["# STG results", "",
         "Paper numbers are 2014, Yan et al. (2023) Plant Methods 19:96, Table 1 (RE %) and Table 2 (r). "
         "Higher r is better, lower RE is better."]


def heading(text):
    lines.extend(["", f"## {text}", ""])


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    lines.append("")


chosen = get("chosen_config.json") or {}
ROLES = ru.report_roles(chosen)

# 1. data check ------------------------------------------------------------------------------------------------------
check = get("data_check.json")
if check:
    heading("1. Data check: ridge on all markers vs the paper's LMM")
    lines.append("Same plants and splits as the paper. If ridge lands close to LMM, your data matches theirs.")
    lines.append("")
    rows = []
    for i, s in enumerate(ru.SPLITS):
        if s in check:
            rows.append([s, num(check[s]["r"]), num(ru.PAPER["LMM"]["r"][i], ".2f"),
                         num(check[s]["re"], ".2f"), num(ru.PAPER["LMM"]["re"][i], ".2f")])
    table(["Split", "Ridge r", "Paper LMM r", "Ridge RE %", "Paper LMM RE %"], rows)

# 2. chosen settings -------------------------------------------------------------------------------------------------
if chosen:
    heading("2. Settings chosen on the validation plants")
    lines.append("best = highest validation r. sparse = highest validation r among settings ending with ≤ "
                 "the sparse limit of open gates (VBS-ML kept 354). Test plants were not used.")
    lines.append("")
    rows = []
    for s in ru.SPLITS:
        for role in ru.ROLES:
            st = chosen.get(s, {}).get(role)
            if st:
                rows.append([s, role, f"{st['stg_lambda']:g}", f"{st['l1_lambda']:g}", st["epochs"],
                             num(st["valid_r"]), num(st["valid_re"], ".2f"), st["gates"]])
    table(["Split", "Setting", "STG λ", "L1 λ", "Epochs", "Valid r", "Valid RE %", "Open gates"], rows)
    if ROLES == ["best"]:
        lines.append("On every split the sparse setting is the same as the best one (or no setting was sparse enough), "
                     "so only 'best' is reported below.")
        lines.append("")

# 3. prediction vs the paper -----------------------------------------------------------------------------------------
heading("3. Prediction on the test plants vs the paper")
lines.append("STG = final model after training on all 90% of training plants, mean ± sd over seeds. "
             "Nothing about these runs was chosen using the test plants.")
lines.append("")
for role in ROLES:
    rows, peek_rows, all_r, all_re = [], [], [], []
    for i, s in enumerate(ru.SPLITS):
        runs = ru.final_runs(R, s, chosen.get(s, {}).get(role))
        if not runs:
            continue
        r = [x["test"]["r"] for x in runs]
        re = [x["test"]["re"] for x in runs]
        gates = [x["final_open_gates"] for x in runs]
        all_r.append(np.mean(r))
        all_re.append(np.mean(re))
        vbs_r, vbs_re = ru.PAPER["VBS-ML (354)"]["r"][i], ru.PAPER["VBS-ML (354)"]["re"][i]
        rows.append([s, pm(r), pm(re, ".2f"), f"{np.mean(gates):.0f}",
                     f"{vbs_r:.2f}", f"{vbs_re:.2f}", f"{ru.PAPER['LMM']['r'][i]:.2f}", f"{ru.PAPER['LMM']['re'][i]:.2f}",
                     f"{sum(x < vbs_re for x in re)}/{len(re)}"])
        peek_rows.append([s, pm([x["test_peek"]["r"] for x in runs]), pm([x["test_peek"]["re"] for x in runs], ".2f"),
                          f"{vbs_r:.2f}", f"{vbs_re:.2f}"])
    if not rows:
        continue
    rows.append(["**Average**", f"**{np.mean(all_r):.3f}**", f"**{np.mean(all_re):.2f}**", "",
                 f"{np.mean(ru.PAPER['VBS-ML (354)']['r']):.2f}", f"{np.mean(ru.PAPER['VBS-ML (354)']['re']):.2f}",
                 f"{np.mean(ru.PAPER['LMM']['r']):.2f}", f"{np.mean(ru.PAPER['LMM']['re']):.2f}", ""])
    lines.append(f"**STG ({role} setting)**")
    lines.append("")
    table(["Split", "STG r", "STG RE %", "STG markers", "VBS-ML r", "VBS-ML RE %", "LMM r", "LMM RE %",
           "Seeds with lower RE than VBS-ML"], rows)
    lines.append(f"Optimistic upper bound for STG ({role}): the lowest test RE seen during training, i.e. picked using "
                 "the test plants. Only for context - the paper does not say how its reported epoch was chosen.")
    lines.append("")
    table(["Split", "STG r", "STG RE %", "VBS-ML r", "VBS-ML RE %"], peek_rows)
others = ", ".join(f"{k} r {np.mean(v['r']):.2f} / RE {np.mean(v['re']):.2f}%" for k, v in ru.PAPER.items())
lines.append(f"Paper averages over the 4 splits: {others}.")

# 4. marker refit test ------------------------------------------------------------------------------------------------
refit = get("refit.json")
if refit:
    heading("4. How good are STG's markers? Refit test")
    lines.append("Ridge regression refitted on the training plants using only K markers, scored on the test plants. "
                 "Cells are r / RE %, averaged over splits (STG: averaged over seeds first). "
                 "\"Random, best 5%\" = 95th percentile of r (5th of RE) over random draws: STG should beat it.")
    lines.append("")
    for role in ROLES:
        # only the splits where this setting has STG runs, so every column is averaged over the same splits
        splits = {s: res for s, res in refit["splits"].items() if role in res["stg"]}
        if not splits:
            continue

        def avg(fn, splits=splits):
            vals = [fn(res) for res in splits.values()]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        def cell(fn_r, fn_re):
            r, re = avg(fn_r), avg(fn_re)
            return "–" if r is None else f"{r:.3f} / {re:.2f}"

        all_cell = cell(lambda x: x["all"]["r"] if x.get("all") else None,
                        lambda x: x["all"]["re"] if x.get("all") else None)
        rows = []
        for k in [str(k) for k in refit["ks"]]:
            if not any(k in res["random"] for res in splits.values()):
                continue #K is larger than the number of markers
            def stg(metric, k=k):
                return lambda x: (float(np.mean([y[metric] for y in x["stg"][role]["per_k"][k]]))
                                  if role in x["stg"] and k in x["stg"][role]["per_k"] else None)

            def rand(metric, q, k=k):
                return lambda x: (float(np.percentile([y[metric] for y in x["random"][k]], q)) if k in x["random"] else None)

            def single(sel, metric, k=k):
                return lambda x: (x[sel][k][metric] if x[sel].get(k) else None)

            rows.append([k, cell(stg("r"), stg("re")), cell(rand("r", 50), rand("re", 50)),
                         cell(rand("r", 95), rand("re", 5)), cell(single("marginal", "r"), single("marginal", "re")),
                         cell(single("lasso", "r"), single("lasso", "re"))])
        lines.append(f"**STG ({role} setting)**, splits {', '.join(splits)} - ridge on all markers: {all_cell}")
        lines.append("")
        table(["K", "STG top-K", "Random (median)", "Random, best 5%", "Top-K by correlation with yield", "LASSO"], rows)

        nat = [n for res in splits.values() if role in res["stg"] for n in res["stg"][role]["natural"]]
        sizes = [n["n_markers"] for n in nat]
        scored = [n for n in nat if n.get("r") is not None]
        if scored:
            lines.append(f"STG's own selection (every open gate, {np.mean(sizes):.0f} markers on average): "
                         f"r {np.mean([n['r'] for n in scored]):.3f} / RE {np.mean([n['re'] for n in scored]):.2f}%, "
                         f"vs random sets of the same size r {np.mean([n['random_r'] for n in scored]):.3f} / "
                         f"RE {np.mean([n['random_re'] for n in scored]):.2f}%.")
        elif sizes and max(sizes) == 0:
            lines.append("STG's own selection is empty: every gate closed in the final runs.")
        elif sizes:
            lines.append(f"STG's own selection has {np.mean(sizes):.0f} markers on average - too many to refit, not scored.")
        lines.append("")

# 5. stability and redundancy -----------------------------------------------------------------------------------------
stab = get("stability.json")
if stab:
    heading("5. Stability and redundancy of STG's markers")
    lines.append("Overlap = share of one run's top-K markers also picked by the other run. LD-aware also counts a "
                 f"marker correlated at |r| ≥ {ru.LD_R} as found. Within split = different seeds; across splits = "
                 "same seed, different training plants.")
    lines.append("")
    for role, res in stab.items():
        rows = []
        for k, w in res["within"].items():
            a, rnd = res["across"][k], res["random"][k]
            rows.append([k, num(w["exact"]), num(w["ld"]), num(a["exact"]), num(a["ld"]), num(rnd["exact"]), num(rnd["ld"])])
        n = res["natural"]
        rows.append([f"own selection (~{n['size_mean']:.0f})", num(n["within"]["exact"]), num(n["within"]["ld"]),
                     num(n["across"]["exact"]), num(n["across"]["ld"]), num(n["random_exact"]), "–"])
        lines.append(f"**STG ({role} setting)**")
        lines.append("")
        table(["K", "Within split", "Within (LD-aware)", "Across splits", "Across (LD-aware)", "Random pair",
               "Random pair (LD-aware)"], rows)

        rows = []
        for k, red in res["redundancy"].items():
            def show(x):
                return "–" if not x else f"{100 * x['near_copy']:.0f}% (mean |r| {x['mean_abs_r']:.2f})"
            rows.append([k, show(red["stg"]), show(red["random"]), show(red["marginal"]), show(red["lasso"])])
        lines.append(f"Redundancy ({role}): share of selected markers that have a near-copy (|r| ≥ {ru.LD_R}) inside "
                     "the same selected set. Lower = fewer wasted slots.")
        lines.append("")
        table(["K", "STG", "Random", "Top-K by correlation", "LASSO"], rows)
        lines.append(f"Most often kept markers: `{R}/top_markers_{role}.csv`")
        lines.append("")

# 6. simulation -------------------------------------------------------------------------------------------------------
sim = get("sim_scores.json")
if sim:
    heading("6. Simulation: does STG find the true markers, and does correlation between markers hurt?")
    lines.append("Simulated yields on the real markers (heritability 0.62). high = true markers have a strongly "
                 "correlated neighbour; low = they don't. Cells are exact / LD-aware.")
    lines.append("")
    rows = []
    for row in sim["summary"]:
        o, t, mg = row["stg_open"], row["stg_top_m"], row["marginal_top_m"]
        rows.append([row["condition"], row["n_true"], row["reps"], num(o["n_selected"], ".0f"),
                     f"{num(o['recall'], '.2f')} / {num(o['recall_ld'], '.2f')}",
                     f"{num(o['precision'], '.2f')} / {num(o['precision_ld'], '.2f')}",
                     f"{num(t['recall'], '.2f')} / {num(t['recall_ld'], '.2f')}",
                     f"{num(mg['recall'], '.2f')} / {num(mg['recall_ld'], '.2f')}", num(row["test_r"])])
    table(["Condition", "True markers (m)", "Reps", "STG open gates", "STG recall", "STG precision",
           "STG top-m recall", "Top-m by correlation recall", "STG test r"], rows)

# 7. notes ------------------------------------------------------------------------------------------------------------
heading("Notes")
lines.extend([
    "- Training follows the paper's stated recipe: 256→128→1 network, dropout after the first layer, MAE loss, "
    "L1 on the network weights, Adam (lr 1e-4, betas 0.9/0.99, weight decay 5e-4), cosine learning-rate schedule, batch 512.",
    "- Assumed (the paper doesn't say): dropout rate 0.5; weight decay is not applied to the STG gates.",
    "- The paper contradicts itself on the network (it also mentions four fully connected layers and three residual blocks).",
    "- The paper reports one run per split and doesn't say how its reported epoch was chosen.",
    "- Settings were chosen separately for each split, using only that split's validation plants.",
])

report = "\n".join(lines) + "\n"
(R / "summary.md").write_text(report)
print(report)
print(f"wrote {R / 'summary.md'}")
