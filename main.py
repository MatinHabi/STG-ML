import model2 as m
from pathlib import Path
import torch

EPOCH = 300

def train():
    if Path("data.pt").exists():
        for e in range(EPOCH):
            m.trainCycle(
                m.X_train, m.Y_train,
                m.device, m.model,e
            )

            m.validCycle(
                m.X_valid,m.Y_valid,
                m.device,m.model,e
            )
        torch.save(m.model.gate.mu.detach().cpu(), f"mu_runs/mu_{m.RUN_SEED}.pt")
    else:
        print("train - data.pt does not exist!\n")

def test():
    if Path("data.pt").exists():
        m.testCycle(
            m.X_test, m.Y_test,
            m.model, m.device)
    else:
        print("test - data.pt does not exist!\n")


def main():
    train()

if __name__ == "__main__":
    main()