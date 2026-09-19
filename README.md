# dnasty

Evolutionary neural architecture search (NAS) for ECG arrhythmia classification, built on PyTorch.
Network components are encoded as *genes*, assembled into a *genome*, expressed as an `nn.Sequential`
and scored by an *estimator*; a *search strategy* decides which genomes to try next.

The project started as MPhys work in 2021. A full review of the code, datasets, search strategies and
search space against the current literature, plus the roadmap that follows from it, is in
[docs/REVIEW.md](docs/REVIEW.md).

## What is implemented

| Component | Available now |
|---|---|
| Search space | `cbam`: chain of `[ConvBlock2d × n → MaxPool2d → CBAM] × cells → Flatten → LinearBlock × linear` on 128×128 single-channel images |
| Search strategy | `RandomSearch` (the baseline every other strategy has to beat); `RegularizedEvolution` (aging evolution, Real et al. 2019) with hyper-parameter, insert-conv and delete-conv mutations and optional one-point crossover at cell boundaries |
| Estimator | `LowFidelityEstimator` (train for a few epochs, best validation accuracy); `SyntheticEstimator` (deterministic training-free landscape for comparing strategies); `CachedEstimator` (memoises on the architecture); `MockEstimator` (random fitness, for tests) |
| Data | PTB-XL 12-lead records with the predefined folds, six label tasks, multi-hot targets and a cached memmap (`ptbxl`); CPSC 2018 single-lead records as raw signals (`cpsc1d`) or Mexican-hat wavelet images (`cpsc2d`); PhysioNet downloads for MIT-BIH, PTB-XL (with subsets) and CPSC 2018 |
| Training | `Trainer` picks the loss from the targets (cross-entropy or `BCEWithLogitsLoss`) and scores `accuracy`, `macro_auroc` or `macro_f1` over the whole validation set; `ECGNet1d` is a 100k-parameter 1D reference model |

Multi-objective selection, zero-cost proxies, predictors and a cell-based search space are the next milestones; see the roadmap in the review.

## Install

Requires Python 3.10 to 3.13. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra data
```

or with pip:

```bash
pip install -e ".[dev,data]"
```

On Intel Macs the resolver pins `torch<2.3` and `numpy<2`, the last builds with wheels for that platform.

## Data

Data is not tracked in git. Either point the tools at an existing directory or download from PhysioNet:

```bash
uv run dnasty data download mitbih --data-dir data
```

```bash
uv run dnasty data download ptbxl --data-dir data
```

PTB-XL is 1.7 GB at 100 Hz; a subset is enough to develop against, and records missing on disk are skipped
by the loader (about 25 kB per record):

```bash
uv run dnasty data download ptbxl --data-dir data --folds 1 2 3 4 5 6 7 8 9 --limit-per-fold 300
```

The data root is resolved as `--data-dir`, then `$DNASTY_DATA_DIR`, then `data_dir` in the config.
The CPSC configs expect `<root>/cpsc_data/test100/*.mat` and `<root>/cpsc_data/reference300.csv`;
`configs/ptbxl.yaml` expects `<root>/ptbxl/` as downloaded.

### PTB-XL

`PTBXLDataset` follows the PTB-XL benchmark (Strodthoff et al. 2021): patient-disjoint folds 1-8 for training,
9 for validation and 10 for testing; tasks `superdiagnostic` (5 classes), `subdiagnostic` (23), `diagnostic`
(44), `form` (19), `rhythm` (12) and `all` (71), with multi-hot targets. On first use the records are read
with `wfdb`, standardised per record and lead, and written once to a float16 memmap under `<root>/ptbxl/cache/`;
later runs and `DataLoader` workers read the memmap only. Multi-label data trains with `BCEWithLogitsLoss` and
is scored with macro-AUROC (mean one-versus-rest AUROC over the classes), the benchmark's metric, or macro-F1.

```bash
uv run python examples/ptbxl_reference.py --config configs/ptbxl.yaml --epochs 8 --repeats 3
```

trains the `ECGNet1d` reference model on the downloaded records and prints per-epoch macro-AUROC, per-class
AUROC and the spread over repeats. On the 2,700-record subset above (2,383 train / 288 validation, superdiagnostic,
Intel CPU, about 5 s per epoch):

| repeats | best macro-AUROC | sd over repeats | after 1 epoch | per-class (CD, HYP, MI, NORM, STTC) |
|---|---|---|---|---|
| 3 × 8 epochs | 0.910 | 0.001 | 0.88 to 0.90 | 0.905, 0.85, 0.91, 0.94, 0.91 |

The published benchmark on the full data (xresnet1d101) is 0.93. The repeat noise here is 0.001 against 0.04 for
the 1-epoch CPSC estimator, so this data gives a usable fitness signal. The `cbam` search space is still 2D, so
PTB-XL is not searchable until the dimension-agnostic search space (roadmap item 3) lands.

## Run a search

```bash
uv run dnasty search --config configs/tiny.yaml --mock-evaluator --seed 0
```

`--estimator low-fidelity` (the default) trains each candidate on the data, and `--strategy` picks the search strategy
(`random` or `regularized_evolution`, overriding `search_strategy` in the config):

```bash
DNASTY_DATA_DIR=datasets uv run dnasty search --config configs/default.yaml --strategy regularized_evolution --seed 0 -v
```

Each run writes `runs/<timestamp>/config.yaml`, `results.jsonl` (the best genome after each generation) and
`evaluated.jsonl` (every evaluated genome in order). Runs with the same config and seed are identical.

## Compare strategies

Every strategy spends the same budget, `population_size × generations` estimator calls, so they can be compared
directly. `dnasty benchmark` runs each strategy at each seed and reports the best fitness found:

```bash
uv run dnasty benchmark --config configs/default.yaml --strategies random regularized_evolution --seeds 0 1 2
```

The default estimator is `synthetic`: a fixed, training-free landscape over the search space (rewards four conv
blocks, kernel sizes near 5, 64 channels and few parameters), which checks that a strategy climbs without costing any
GPU time. With the default config (100 evaluations, 3 seeds) it gives:

| strategy | best mean | std | min | max | unique archs | time/s |
|---|---|---|---|---|---|---|
| random | 0.7994 | 0.0141 | 0.7864 | 0.8190 | 100.0 | 0.81 |
| regularized_evolution | 0.8930 | 0.0295 | 0.8580 | 0.9301 | 94.3 | 0.28 |

`--estimator low-fidelity` runs the same comparison with real training (one shared data split and a shared result
cache, so each architecture is trained once). The run directory `runs/bench-<timestamp>/` holds `results.jsonl` with
the best-so-far curve of every run and `summary.json`. The wall time of a synthetic run is the strategy's own
overhead, which is the baseline for the planned Mojo port.

On the 100-record CPSC subset (`configs/bench_cpsc.yaml`: 40 evaluations, 1 training epoch, 60 train / 40 validation
records, Intel CPU) the two strategies are indistinguishable:

```bash
DNASTY_DATA_DIR=datasets uv run dnasty benchmark --config configs/bench_cpsc.yaml --estimator low-fidelity --seeds 0 1 2
```

| strategy | best mean | std | min | max | time/s |
|---|---|---|---|---|---|
| random | 0.3167 | 0.0118 | 0.3000 | 0.3250 | 132 |
| regularized_evolution | 0.3083 | 0.0118 | 0.3000 | 0.3250 | 29 |

That is expected: with 40 validation records the accuracy resolution is 0.025, the majority-class baseline is 0.275,
and re-training the same genome four times moves its fitness by about 0.04 (as much as the spread between
architectures). One epoch on 60 records is three gradient steps, so the estimator cannot rank architectures on this
subset and the "best" genome is just the largest of 40 noisy draws. A meaningful comparison needs the full CPSC 2018
set or PTB-XL, more epochs, and a finer metric (macro-F1 or AUROC): roadmap item 4 in the review. The evolution run
is faster because its initial population is the same ten genomes random search evaluated for that seed (served from
the cache) and its children are mostly small models.

From Python:

```python
from dnasty import Config, DataModule, LowFidelityEstimator, build_strategy, seed_everything

config = Config.from_file("configs/default.yaml").nas
seed_everything(config.seed)
datamodule = DataModule.from_config(config)          # one seeded split, shared everywhere
estimator = LowFidelityEstimator(config, datamodule=datamodule)
search = build_strategy(config, estimator=estimator, name="regularized_evolution")
search.fit()
print(search.fittest_genome)
```

## Develop

```bash
uv run pytest -q
```

```bash
uv run ruff check src tests
```

Tests that need the CPSC files are marked `data` and skip when the files are absent. `uv build` produces the wheel and sdist.

## Layout

```
src/dnasty/
  data/             CPSC and PTB-XL datasets, DataModule (seeded or predefined splits), PhysioNet downloads
  defaults/         Trainer (CE or BCE, accuracy / macro-AUROC / macro-F1) and reference models (2D CBAM, ECGNet1d)
  estimators/       fitness estimators
  search_space/     common building blocks + the cbam space (genes, genome, components)
  search_strategies/ RandomSearch, RegularizedEvolution, mutation/crossover operators
  benchmark.py      strategy comparison harness (dnasty benchmark)
  utils/            Config (YAML/JSON), seeding, metrics (AUROC, F1), wavelets
configs/            default.yaml, tiny.yaml, bench_cpsc.yaml, ptbxl.yaml
docs/REVIEW.md      review and roadmap
examples/           fitness-vs-trained-score correlation; PTB-XL reference training
```

## License

MIT, see [LICENSE](LICENSE).
