"""Build the train-only calibrated statistics consumed by C01 E training."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Statistics already exist: {args.output}")
    import torch
    from afce_all11.calibrated_codec import fit_statistics
    from afce_all11.evidence import EvidenceData
    from afce_all11.train_calibrated import atomic_save

    torch.set_num_threads(4)
    train = EvidenceData(args.data, args.evidence, "train", "legacy-none")
    if (len(train), len(train.episodes)) != (469908, 990):
        raise ValueError("Expected the C01 990-episode training split")
    contract = {"data": str(args.data.resolve()), "evidence": str(args.evidence.resolve()),
                "split": "episodes0-89", "frames": len(train), "version": "calibrated_v2"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({"stats": fit_statistics(train), "contract": contract}, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
