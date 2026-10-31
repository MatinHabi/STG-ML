import model2 as m
from pathlib import Path
import torch

EPOCH = 1500
EVAL_EVERY = 10      # epochs between whole-set evaluations
MATCH_K = 354        # VBS-ML's marker count for 2014 (paper Tables 1-2)

def fmt(tag, d):
    return f"{tag}: r={d['r']:.3f} RE={d['re']:.2f}% MSE={d['mse']:.3f} gates={d['gates']}"

def train():
    best = {"valid_r": -1.0}     # honest: epoch picked on valid
    best_vbs = {"re": 1e9}       # VBS protocol: epoch picked on test RE (what their mre.py does)
    for e in range(EPOCH):
        m.trainCycle(
            m.X_train, m.Y_train,
            m.device, m.model,e
        )

        if (e + 1) % EVAL_EVERY == 0:
            v = m.evaluate(m.X_valid, m.Y_valid)
            t = m.evaluate(m.X_test, m.Y_test)
            t_k = m.evaluate(m.X_test, m.Y_test, top_k=MATCH_K)
            print(f"EVAL: STG_LAMBDA: {m.STG_LAMBDA} || epoch: {e} || {fmt('valid', v)} || {fmt('test', t)} || {fmt(f'test@{MATCH_K}', t_k)}")
            if v["r"] > best["valid_r"]:
                best = {"valid_r": v["r"], "epoch": e, "test": t, "test_k": t_k,
                        "mu": m.model.gate.mu.detach().cpu().clone()}
            if t["re"] < best_vbs["re"]:
                best_vbs = {**t, "epoch": e}

    print(f"\nFINAL {m.FILE} STG_LAMBDA={m.STG_LAMBDA}")
    print(f"  honest       (epoch {best['epoch']}, picked by valid r)  {fmt('test', best['test'])} || {fmt(f'test@{MATCH_K}', best['test_k'])}")
    print(f"  VBS protocol (epoch {best_vbs['epoch']}, picked by test RE) {fmt('test', best_vbs)}")

    run_name = f"{Path(m.FILE).stem}_seed{m.RUN_SEED}_stg{m.STG_LAMBDA}_l1{m.L1_LAMBDA}_lr{m.ADAM_LR}_sig{m.SIGMA}"
    Path("mu_runs").mkdir(exist_ok=True)
    torch.save(best["mu"], f"mu_runs/mu_{run_name}.pt")   # gates from the best-valid epoch

def test():
    m.testCycle(
        m.X_test, m.Y_test,
        m.model, m.device)

def main():
    train()

if __name__ == "__main__":
    main()