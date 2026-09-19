"""Train the ECGNet1d reference on PTB-XL and report macro-AUROC.

Checks that the data pipeline yields a real fitness signal: the validation
macro-AUROC should rise well above 0.5 within a few epochs, and re-training
the same model with different seeds (``--repeats``) shows how much of the
score is noise. Records missing on disk are skipped, so a subset download
is enough::

    dnasty data download ptbxl --data-dir data --folds 1 2 3 4 5 6 7 8 9 --limit-per-fold 300
    python examples/ptbxl_reference.py --config configs/ptbxl.yaml --epochs 8 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from dnasty import Config, DataModule, Trainer, seed_everything
from dnasty.defaults import ECGNet1d
from dnasty.utils.metrics import per_class_auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--repeats", type=int, default=1, help="re-train with new seeds"
    )
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    config = Config.from_file(args.config).nas
    epochs = args.epochs or config.train.epochs
    seed_everything(config.seed)
    t0 = time.perf_counter()
    dm = DataModule.from_config(config, args.data_dir)
    train, val = dm.trainset, dm.valset
    print(
        f"data ready in {time.perf_counter() - t0:.1f}s: {len(train)} train, "
        f"{len(val)} val records, task {train.task}, classes {train.classes}"
    )
    print(f"train positives per class: {train.class_counts()}")
    print(f"val positives per class:   {val.class_counts()}")

    trainer = Trainer(config, dm)
    finals: list[float] = []
    for repeat in range(args.repeats):
        seed_everything(config.seed + repeat)
        model = ECGNet1d(dm.in_channels, dm.num_classes, width=args.width)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nrepeat {repeat}: ECGNet1d width {args.width}, {n_params} params")
        t0 = time.perf_counter()
        result = trainer.fit_detailed(model, epochs)
        for i, epoch in enumerate(result.epochs, start=1):
            print(
                f"  epoch {i:2d}  train loss {epoch.train_loss:.4f}  "
                f"val loss {epoch.val_loss:.4f}  val {result.metric} {epoch.val_score:.4f}"
            )
        print(
            f"  best {result.metric} {result.best_val_score:.4f} "
            f"in {time.perf_counter() - t0:.0f}s"
        )
        finals.append(result.best_val_score)

        model.eval()
        with torch.inference_mode():
            outputs = torch.cat(
                [model(x.to(trainer.device)).cpu() for x, _ in dm.val_loader]
            )
            targets = torch.cat([y for _, y in dm.val_loader])
        per_class = per_class_auroc(outputs, targets)
        print(
            "  per-class AUROC: "
            + ", ".join(
                f"{name} {value:.3f}"
                for name, value in zip(val.classes, per_class, strict=True)
            )
        )

    if args.repeats > 1:
        print(
            f"\nbest {trainer.metric} over {args.repeats} repeats: "
            f"mean {statistics.fmean(finals):.4f} sd {statistics.pstdev(finals):.4f}"
        )


if __name__ == "__main__":
    main()
