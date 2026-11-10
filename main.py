# Trains one STG model. Two modes (set with --mode, see model3.py for all the options):
#
#   select  tuning run on data_split??.pt. Trains on X_train and only ever scores the VALIDATION plants.
#           select_config.py later compares these runs to choose STG_LAMBDA / L1_LAMBDA / epochs.
#
#   fixed   final run on data_split??_full.pt (all 90% of VBS's training plants, like the paper).
#           Trains for exactly --epochs and reports the FINAL model on the test plants. Nothing is chosen
#           using the test plants. The test set is also scored every EVAL_EVERY epochs, but only to report
#           an optimistic upper bound (best test RE seen during training) - it never changes the model.
#
# Each run saves <out_dir>/runs/<run name>.json (all numbers) and <out_dir>/mu/mu_<run name>.pt (the gates).
import time
import model3 as m
import torch
from results_utils import write_json

def fmt(tag,d):
    return f"{tag}: r={d['r']:.3f} RE={d['re']:.2f}% MSE={d['mse']:.3f} gates={d['gates']}"

def is_eval_epoch(e):
    return (e + 1) % m.EVAL_EVERY == 0 or e + 1 == m.EPOCHS #always evaluate after the last epoch

def log(e, loss, text):
    print(f"epoch {e + 1}/{m.EPOCHS} || lr {m.scheduler.get_last_lr()[0]:.2e} || train_loss {loss:.4f} || {text}", flush=True)


def tune():
    curve = []
    best = {"r": -2.0}

    for e in range(m.EPOCHS):
        loss = m.training()

        if is_eval_epoch(e):
            v = m.evaluate(m.X_valid, m.Y_valid)
            curve.append({"epoch": e + 1, "train_loss": loss, **v})
            log(e, loss, fmt('valid', v))

            if v["r"] > best["r"]:
                best = {**v, "epoch": e + 1} #only reported (shows if the model peaked early); select_config.py uses the FINAL model

    final = m.evaluate(m.X_valid, m.Y_valid)
    print(f"\nFINAL (tuning) {m.RUN_NAME}")
    print(f"  final model                    {fmt('valid', final)}")
    print(f"  best checkpoint (epoch {best['epoch']})   {fmt('valid', best)}")
    return {"final_valid": final, "best_valid": best, "curve": curve}


def final_run():
    curve = []
    peek = {"re": float("inf")}

    for e in range(m.EPOCHS):
        loss = m.training()

        if is_eval_epoch(e):
            t = m.evaluate(m.X_test, m.Y_test)
            curve.append({"epoch": e + 1, "train_loss": loss, **t})
            log(e, loss, fmt('test', t))

            if t["re"] < peek["re"]:
                peek = {**t, "epoch": e + 1} #**t takes every key/value pair in t and inserts it into the new dictionary

    final = m.evaluate(m.X_test, m.Y_test)
    print(f"\nFINAL {m.RUN_NAME}")
    print(f"  final model (headline number, test never used for any choice)   {fmt('test', final)}")
    print(f"  lowest test RE seen (epoch {peek['epoch']}, optimistic upper bound)  {fmt('test', peek)}")
    return {"test": final, "test_peek": peek, "curve": curve}


def main():
    start = time.time()
    out = tune() if m.MODE == "select" else final_run()

    m.MU_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.model.gates.mu.detach().cpu().clone(), m.MU_PATH) #mu at the END of training (the finished selection)

    write_json(m.RUN_JSON, {
        "run": m.RUN_NAME,
        "config": m.CONFIG,
        "device": m.device,
        "final_open_gates": m.open_gates(),
        "mu_path": str(m.MU_PATH),
        "minutes": round((time.time() - start) / 60, 2),
        **out,
    })
    print(f"saved {m.RUN_JSON}")

if __name__ == "__main__":
    main()
