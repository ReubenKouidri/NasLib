"""Command-line interface: ``dnasty search``, ``dnasty benchmark`` and
``dnasty data download``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from dnasty.utils import Config, seed_everything

ESTIMATORS = ("low-fidelity", "synthetic", "mock")


def _load_config(path: str) -> Config:
    config = Config.from_file(path)
    return config.nas if "nas" in config else config


def _estimator_factory(kind: str, config: Config, data_dir: str | None):
    """Returns ``seed -> Estimator``.

    ``low-fidelity`` shares one data split (seeded by the config) and one
    result cache across all runs so that an architecture is trained once;
    ``synthetic`` uses one landscape (seeded by the config) so that runs with
    different seeds search the same problem.
    """
    from dnasty.estimators import (
        CachedEstimator,
        LowFidelityEstimator,
        MockEstimator,
        SyntheticEstimator,
    )

    landscape_seed = int(config.get("seed", 0))
    if kind == "synthetic":
        return lambda seed: SyntheticEstimator(seed=landscape_seed)
    if kind == "mock":
        return lambda seed: MockEstimator(seed=seed)
    if kind == "low-fidelity":
        from dnasty.data import DataModule

        datamodule = DataModule.from_config(config, data_dir)
        shared = CachedEstimator(LowFidelityEstimator(config, datamodule=datamodule))
        return lambda seed: shared
    raise ValueError(f"Unknown estimator '{kind}'; choose from {ESTIMATORS}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def cmd_search(args: argparse.Namespace) -> int:
    from dnasty.search_strategies import build_strategy

    config = _load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    seed_everything(seed)

    kind = "mock" if args.mock_evaluator else args.estimator
    estimator = _estimator_factory(kind, config, args.data_dir)(seed)
    strategy = build_strategy(config, estimator=estimator, name=args.strategy)
    strategy.fit()

    run_dir = Path(args.run_dir) / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = config.to_dict()
    cfg_dict["seed"] = seed
    cfg_dict["search_strategy"] = strategy.name
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    with (run_dir / "results.jsonl").open("w") as f:
        for gen, genome in enumerate(strategy.history):
            record = {"generation": gen, **genome.to_dict()}
            f.write(json.dumps(record) + "\n")
    with (run_dir / "evaluated.jsonl").open("w") as f:
        for index, genome in enumerate(strategy.evaluated):
            f.write(json.dumps({"evaluation": index, **genome.to_dict()}) + "\n")
    best = strategy.fittest_genome
    print(f"Run written to {run_dir}")
    print(f"Best fitness {best.fitness:.4f} with {best.num_params} params")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from dnasty.benchmark import format_summary, run_benchmark

    config = _load_config(args.config)
    factory = _estimator_factory(args.estimator, config, args.data_dir)
    run_dir = Path(args.run_dir) / f"bench-{_timestamp()}"
    _, summary = run_benchmark(
        config, args.strategies, args.seeds, factory, run_dir=run_dir
    )
    print(format_summary(summary))
    print(f"Results written to {run_dir}")
    return 0


def cmd_data_download(args: argparse.Namespace) -> int:
    from dnasty.data.download import download

    options = {}
    if args.name == "ptbxl":
        options = dict(
            sampling_rate=args.sampling_rate,
            folds=args.folds,
            limit_per_fold=args.limit_per_fold,
        )
    elif args.folds or args.limit_per_fold or args.sampling_rate != 100:
        print("--folds, --limit-per-fold and --sampling-rate apply to ptbxl only")
        return 2
    target = download(args.name, args.data_dir, **options)
    print(f"Downloaded {args.name} to {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnasty")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="run an architecture search")
    search.add_argument("--config", required=True, help="YAML or JSON config")
    search.add_argument("--seed", type=int, default=None)
    search.add_argument(
        "--strategy", default=None, help="overrides search_strategy in the config"
    )
    search.add_argument(
        "--estimator",
        choices=ESTIMATORS,
        default="low-fidelity",
        help="low-fidelity trains each candidate; synthetic and mock do not",
    )
    search.add_argument("--data-dir", default=None, help="overrides DNASTY_DATA_DIR")
    search.add_argument("--run-dir", default="runs")
    search.add_argument(
        "--mock-evaluator",
        action="store_true",
        help="alias for --estimator mock",
    )
    search.set_defaults(func=cmd_search)

    bench = sub.add_parser(
        "benchmark", help="compare strategies at a fixed budget over several seeds"
    )
    bench.add_argument("--config", required=True, help="YAML or JSON config")
    bench.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "regularized_evolution"],
    )
    bench.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    bench.add_argument("--estimator", choices=ESTIMATORS, default="synthetic")
    bench.add_argument("--data-dir", default=None, help="overrides DNASTY_DATA_DIR")
    bench.add_argument("--run-dir", default="runs")
    bench.set_defaults(func=cmd_benchmark)

    data = sub.add_parser("data", help="dataset utilities")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    dl = data_sub.add_parser("download", help="download a dataset from PhysioNet")
    dl.add_argument("name", choices=["mitbih", "ptbxl", "cpsc2018"])
    dl.add_argument("--data-dir", default="data")
    dl.add_argument(
        "--sampling-rate", type=int, choices=[100, 500], default=100, help="ptbxl only"
    )
    dl.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="ptbxl only: strat_fold values",
    )
    dl.add_argument(
        "--limit-per-fold",
        type=int,
        default=None,
        help="ptbxl only: at most this many records per fold (~25 kB each at 100 Hz)",
    )
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
