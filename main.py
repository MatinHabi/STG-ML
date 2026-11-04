import model3 as m
from pathlib import Path
import torch
import copy

EPOCH = 150
EVAL_EVERY = 10 
MATCH_K = 354 #number of markers picked by VBS on teh 2014 data

def fmt(tag,d):
    return f"{tag}: r={d['r']:.3f} RE={d['re']:.2f}% MSE={d['mse']:.3f} gates={d['gates']}"

def train():
    best = {"valid_r": -1.0}
    best_vbs = {"re": 1e9}

    for e in range(EPOCH):
        m.training(e)

        if (e+1) % EVAL_EVERY == 0 :
            v = m.evaluate(m.X_valid,m.Y_valid)
            t = m.evaluate(m.X_test, m.Y_test)
            t_k = m.evaluate(m.X_test, m.Y_test, MATCH_K)
            print(f'EVAL -- STG_LAMBDA: {m.STG_LAMBDA} || epoch: {e} || {fmt('valid', v)} || {fmt('test', t)} || {fmt(f'test@{MATCH_K}', t_k)}')

            if v["r"] > best['valid_r']:
                best = {'valid_r': v["r"] , "epoch": e, "test": t, "test_k" : t_k, "mu": m.model.gates.mu.detach().cpu().clone(), "weights": copy.deepcopy(m.model.state_dict())}

            if t["re"] < best_vbs["re"]:
                best_vbs = {**t, "epoch":e} #**t takes every key/value pair in t and inserts it into the new dictionaryy

    print(f'\nFINAL {m.FILE} STG_LAMBDA = {m.STG_LAMBDA}')
    print(f"unbiased (epoch {best['epoch']}, chosen by best VALID r -> test set never used for selection)  {fmt('test', best['test'])} || {fmt(f'test@{MATCH_K}', best['test_k'])}")
    print(f"VBS-style (epoch {best_vbs['epoch']}, chosen by lowest TEST RE -> optimistic, matches paper's protocol)  {fmt('test', best_vbs)}")

    run_name = f"{Path(m.FILE).stem}_seed{m.RUN_SEED}_stg{m.STG_LAMBDA}_l1{m.L1_LAMBDA}_lr{m.ADAM_LR}_sig{m.SIGMA}"
    Path("mu_runs").mkdir(exist_ok=True)
    torch.save(best["mu"], f"mu_runs/mu_{run_name}.pt")

    return best


def test(best):
    m.model.loead_state_dict(best["weights"])
    t = m.evaluate(m.X_test, m.Y_test)
    t_k = m.evaluate(m.X_test, m.Y_test, MATCH_K)
    print(f"TEST-- {fmt('test', t)} || {fmt(f'test@{MATCH_K}', t_k)}")
    m.testing()

def main():
    best = train()
    test(best)

if __name__ == "__main__":
    main()