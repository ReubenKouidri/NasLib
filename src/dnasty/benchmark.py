"""Benchmark harness: compare search strategies at a fixed evaluation budget.

Every (strategy, seed) pair runs with the same config, the same budget
(``population_size * generations`` estimator calls) and the same estimator
factory. Results are best-so-far curves per evaluation, so strategies can be
compared at any budget up to the full one, and a summary over seeds.

Wall time is recorded per run; with a training-free estimator it measures the
strategy's own overhead, which is the baseline for the planned Mojo port.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from dnasty.estimators import Estimator
from dnasty.search_strategies import build_strategy
from dnasty.utils import Config, seed_everything

EstimatorFactory = Callable[[int], Estimator]


@dataclass
class RunResult:
    """One strategy run at one seed."""

    strategy: str
    seed: int
    evaluations: int
    unique_architectures: int
    wall_time: float
    curve: list[float]
    best: dict[str, Any]

    @property
    def final(self) -> float:
        return self.curve[-1] if self.curve else math.nan


def run_once(
    config: Config,
    strategy_name: str,
    seed: int,
    estimator_factory: EstimatorFactory,
) -> RunResult:
    seed_everything(seed)
    strategy = build_strategy(
        config, estimator=estimator_factory(seed), name=strategy_name
    )
    start = time.perf_counter()
    strategy.fit()
    wall = time.perf_counter() - start

    curve: list[float] = []
    best = -math.inf
    for genome in strategy.evaluated:
        best = max(best, genome.fitness)
        curve.append(best)
    unique = len({g.arch_key() for g in strategy.evaluated})
    return RunResult(
        strategy=strategy_name,
        seed=seed,
        evaluations=len(strategy.evaluated),
        unique_architectures=unique,
        wall_time=wall,
        curve=curve,
        best=strategy.best_so_far.to_dict(),
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarise(results: Sequence[RunResult]) -> dict[str, dict[str, Any]]:
    """Per-strategy statistics over seeds.

    ``curve_mean`` is the mean best-so-far after each evaluation, truncated to
    the shortest run, so strategies can be compared at smaller budgets too.
    """
    summary: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(r.strategy for r in results):
        runs = [r for r in results if r.strategy == name]
        finals = [r.final for r in runs]
        mean, std = _mean_std(finals)
        length = min(len(r.curve) for r in runs)
        curve_mean = [statistics.fmean(r.curve[i] for r in runs) for i in range(length)]
        summary[name] = {
            "seeds": [r.seed for r in runs],
            "evaluations": runs[0].evaluations,
            "final_mean": mean,
            "final_std": std,
            "final_min": min(finals),
            "final_max": max(finals),
            "unique_architectures_mean": statistics.fmean(
                r.unique_architectures for r in runs
            ),
            "wall_time_mean": statistics.fmean(r.wall_time for r in runs),
            "curve_mean": curve_mean,
        }
    return summary


def format_summary(summary: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'strategy':<24}{'seeds':>6}{'evals':>7}{'best mean':>11}"
        f"{'std':>8}{'min':>8}{'max':>8}{'unique':>8}{'time/s':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, s in summary.items():
        lines.append(
            f"{name:<24}{len(s['seeds']):>6}{s['evaluations']:>7}"
            f"{s['final_mean']:>11.4f}{s['final_std']:>8.4f}"
            f"{s['final_min']:>8.4f}{s['final_max']:>8.4f}"
            f"{s['unique_architectures_mean']:>8.1f}{s['wall_time_mean']:>9.2f}"
        )
    return "\n".join(lines)


def run_benchmark(
    config: Config,
    strategies: Sequence[str],
    seeds: Sequence[int],
    estimator_factory: EstimatorFactory,
    run_dir: str | Path | None = None,
) -> tuple[list[RunResult], dict[str, dict[str, Any]]]:
    """Run every strategy at every seed and summarise.

    When ``run_dir`` is given, writes ``config.yaml``, ``results.jsonl`` (one
    line per run, including its curve and best genome) and ``summary.json``.
    """
    results = [
        run_once(config, name, seed, estimator_factory)
        for name in strategies
        for seed in seeds
    ]
    summary = summarise(results)

    if run_dir is not None:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False)
        )
        with (out / "results.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, summary
