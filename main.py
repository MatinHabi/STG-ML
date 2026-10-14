import model2 as m
from pathlib import Path
import torch

EPOCH = 1500

def train():
    for e in range(EPOCH):
        m.trainCycle(
            m.X_train, m.Y_train,
            m.device, m.model,e
        )

        m.validCycle(
            m.X_valid,m.Y_valid,
            m.device,m.model,e
        )
    run_name = f"seed{m.RUN_SEED}_stg{m.STG_LAMBDA}_l1{m.L1_LAMBDA}_lr{m.ADAM_LR}_sig{m.SIGMA}"
    torch.save(m.model.gate.mu.detach().cpu(), f"mu_runs/mu_{run_name}.pt")

def test():
    m.testCycle(
        m.X_test, m.Y_test,
        m.model, m.device)

def main():
    train()

if __name__ == "__main__":
    main()