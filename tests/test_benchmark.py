import json

from dnasty.benchmark import format_summary, run_benchmark, run_once
from dnasty.estimators import SyntheticEstimator


def factory(seed):
    return SyntheticEstimator(seed=0)


def test_run_once_curve_is_best_so_far(tiny_config):
    result = run_once(tiny_config, "random", 0, factory)
    assert result.evaluations == tiny_config.population_size * tiny_config.generations
    assert len(result.curve) == result.evaluations
    assert result.curve == sorted(result.curve)
    assert result.best["fitness"] == result.final
    assert 1 <= result.unique_architectures <= result.evaluations


def test_run_benchmark_writes_files_and_summary(tiny_config, tmp_path):
    results, summary = run_benchmark(
        tiny_config,
        ["random", "regularized_evolution"],
        [0, 1],
        factory,
        run_dir=tmp_path / "bench",
    )
    assert len(results) == 4
    assert set(summary) == {"random", "regularized_evolution"}
    for stats in summary.values():
        assert stats["seeds"] == [0, 1]
        assert stats["final_min"] <= stats["final_mean"] <= stats["final_max"]
        assert len(stats["curve_mean"]) == stats["evaluations"]

    lines = (tmp_path / "bench" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["strategy"] == "random"
    assert json.loads((tmp_path / "bench" / "summary.json").read_text()) == summary
    assert (tmp_path / "bench" / "config.yaml").exists()

    table = format_summary(summary)
    assert "regularized_evolution" in table
    assert table.count("\n") == 3


def test_same_seed_gives_same_result(tiny_config):
    a = run_once(tiny_config, "regularized_evolution", 5, factory)
    b = run_once(tiny_config, "regularized_evolution", 5, factory)
    assert a.curve == b.curve
    assert a.best == b.best
