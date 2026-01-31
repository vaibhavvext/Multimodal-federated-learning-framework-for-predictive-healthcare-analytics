import argparse
from framework.registry_ops import load_registry
from framework.trainer import run_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="weighted_fedavg", choices=["fedavg", "weighted_fedavg"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reg = load_registry("data/hospitals.json")
    label_col = reg["label_col"]
    hospitals = reg["hospitals"]

    run_training(
        hospitals=hospitals,
        label_col=label_col,
        algo=args.algo,
        rounds=args.rounds,
        seed=args.seed,
        state_dir="state",
        outputs_dir="outputs"
    )

if __name__ == "__main__":
    main()
