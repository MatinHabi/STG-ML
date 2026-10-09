import model2 as m
from pathlib import Path

EPOCH = 100

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

def test():
    if Path("data.pt").exists():
        m.testCycle(
            m.X_test, m.Y_test,
            m.device, m.model)


def main():
    train()

if __name__ == "__main__":
    main()