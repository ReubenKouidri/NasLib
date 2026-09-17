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
| Search strategy | `RandomSearch` (the baseline every other strategy has to beat) |
| Estimator | `LowFidelityEstimator` (train for a few epochs, best validation accuracy), `MockEstimator` (random fitness, for tests and benchmarks) |
| Data | CPSC 2018 single-lead records as raw signals (`cpsc1d`) or Mexican-hat wavelet images (`cpsc2d`); PhysioNet downloads for MIT-BIH, PTB-XL and CPSC 2018 |

Evolutionary operators (mutation, crossover, aging selection) are the next milestone; see the roadmap in the review.

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

The data root is resolved as `--data-dir`, then `$DNASTY_DATA_DIR`, then `data_dir` in the config.
The bundled configs expect `<root>/cpsc_data/test100/*.mat` and `<root>/cpsc_data/reference300.csv`.

## Run a search

```bash
uv run dnasty search --config configs/tiny.yaml --mock-evaluator --seed 0
```

drops `--mock-evaluator` to train each candidate on the data:

```bash
DNASTY_DATA_DIR=datasets uv run dnasty search --config configs/default.yaml --seed 0 -v
```

Each run writes `runs/<timestamp>/config.yaml` and `results.jsonl` (one line per generation with the best genome, its
fitness and parameter count). Runs with the same config and seed are identical.

From Python:

```python
from dnasty import Config, DataModule, LowFidelityEstimator, RandomSearch, seed_everything

config = Config.from_file("configs/default.yaml").nas
seed_everything(config.seed)
datamodule = DataModule.from_config(config)          # one seeded split, shared everywhere
search = RandomSearch(config, estimator=LowFidelityEstimator(config, datamodule=datamodule))
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
  data/             datasets, seeded splits, PhysioNet downloads
  defaults/         Trainer and a hand-designed reference model
  estimators/       fitness estimators
  search_space/     common building blocks + the cbam space (genes, genome, components)
  search_strategies/
  utils/            Config (YAML/JSON), seeding, metrics, wavelets
configs/            default.yaml, tiny.yaml
docs/REVIEW.md      review and roadmap
examples/           fitness-vs-trained-score correlation script
```

## License

MIT, see [LICENSE](LICENSE).
