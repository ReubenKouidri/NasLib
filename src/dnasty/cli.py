"""Command-line interface: ``dnasty search`` and ``dnasty data download``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from dnasty.utils import Config, seed_everything


def _load_config(path: str) -> Config:
    config = Config.from_file(path)
    return config.nas if "nas" in config else config


def cmd_search(args: argparse.Namespace) -> int:
    from dnasty.estimators import Estimator, LowFidelityEstimator, MockEstimator
    from dnasty.search_strategies import RandomSearch

    config = _load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    seed_everything(seed)

    estimator: Estimator
    if args.mock_evaluator:
        estimator = MockEstimator(seed=seed)
    else:
        from dnasty.data import DataModule

        datamodule = DataModule.from_config(config, args.data_dir)
        estimator = LowFidelityEstimator(config, datamodule=datamodule)

    strategy = RandomSearch(config, estimator=estimator)
    strategy.fit()

    run_dir = Path(args.run_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = config.to_dict()
    cfg_dict["seed"] = seed
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    with (run_dir / "results.jsonl").open("w") as f:
        for gen, genome in enumerate(strategy.history):
            record = {"generation": gen, **genome.to_dict()}
            f.write(json.dumps(record) + "\n")
    best = strategy.fittest_genome
    print(f"Run written to {run_dir}")
    print(f"Best fitness {best.fitness:.4f} with {best.num_params} params")
    return 0


def cmd_data_download(args: argparse.Namespace) -> int:
    from dnasty.data.download import download

    target = download(args.name, args.data_dir)
    print(f"Downloaded {args.name} to {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnasty")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="run an architecture search")
    search.add_argument("--config", required=True, help="YAML or JSON config")
    search.add_argument("--seed", type=int, default=None)
    search.add_argument("--data-dir", default=None, help="overrides DNASTY_DATA_DIR")
    search.add_argument("--run-dir", default="runs")
    search.add_argument(
        "--mock-evaluator",
        action="store_true",
        help="random fitness instead of training (smoke tests, benchmarks)",
    )
    search.set_defaults(func=cmd_search)

    data = sub.add_parser("data", help="dataset utilities")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    dl = data_sub.add_parser("download", help="download a dataset from PhysioNet")
    dl.add_argument("name", choices=["mitbih", "ptbxl", "cpsc2018"])
    dl.add_argument("--data-dir", default="data")
    dl.set_defaults(func=cmd_data_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
