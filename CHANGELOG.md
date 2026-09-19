# Changelog

## Unreleased

### Added
- PTB-XL: `PTBXLDataset` (predefined folds, tasks `superdiagnostic`/`subdiagnostic`/`diagnostic`/`form`/
  `rhythm`/`all`, multi-hot targets, per-record standardisation, float16 memmap cache under `<root>/cache/`);
  `DataModule` accepts predefined train/validation sets; `configs/ptbxl.yaml`; `examples/ptbxl_reference.py`.
- Metrics `macro_auroc`, `macro_f1`, `accuracy` in `dnasty.utils.metrics`; `Trainer` chooses the loss from the
  targets (`BCEWithLogitsLoss` for multi-hot) and the score from `config.metric`; `ECGNet1d` reference model.
- `dnasty data download ptbxl --folds ... --limit-per-fold N --sampling-rate {100,500}` for subsets.
- `RegularizedEvolution` (aging evolution, Real et al. 2019) with tournament selection and a `evolution:` config
  section; operators in `dnasty.search_strategies.operators`: hyper-parameter step, insert-conv and delete-conv
  mutations, one-point crossover at cell boundaries.
- `dnasty benchmark`: runs each strategy at each seed with the same evaluation budget and writes best-so-far curves
  and a summary; `SyntheticEstimator` (deterministic, training-free landscape) and `CachedEstimator`.
- `configs/bench_cpsc.yaml` and the first low-fidelity benchmark result on the CPSC subset (README).
- `Genome.arch_key()`, `to_sequence()`, `spawn()`; `build_strategy()` registry; `dnasty search --strategy/--estimator`
  and `evaluated.jsonl` per run.

### Fixed
- `dnasty data download ptbxl|cpsc2018` built URLs with the version twice (`ptb-xl/1.0.3/1.0.3`) and failed;
  `wfdb.dl_database` resolves the version itself.
- `CBAMGene.mutate` changed the sub-genes, which the next `sync` overwrote; it now mutates the CBAM exons.
- `.gitignore` no longer ignores `src/dnasty/data/`.

### Changed
- `FitResult.best_val_acc` / `EpochResult.val_acc` are now `best_val_score` / `val_score` (old names kept as
  read-only aliases); `Trainer.fit` returns the configured metric, not necessarily accuracy.
- `LinearBlockGene.mutate` changes one feature (dropout or width) instead of both.

## 0.1.0 (unreleased)

First packaged version. See `docs/REVIEW.md` for the full review that motivated these changes.

### Fixed
- Final layer no longer applies `Softmax` before `CrossEntropyLoss` (double softmax).
- Validation accuracy is now `correct / samples`, not `correct / batches`.
- Train/validation split is seeded and shared between the search estimator and post-search training.
- `ConvBlock2d` honours the `batch_norm` flag; block order is conv → BN → activation.
- Spatial attention gate ends with the sigmoid (no BatchNorm after it).
- CPSC labels are joined on the recording id instead of file order; missing secondary labels are `-1`, not class 0.
- Dataset items are always tensors.

### Changed
- `src/` layout, `dnasty.utils` (was `my_utils`), `dnasty.data` (was top-level `datasets`).
- Configuration is YAML (JSON still accepted) with `seed`, `data_dir` and `image_dims`; no absolute paths.
- Data is no longer tracked in git; use `dnasty data download {mitbih,ptbxl,cpsc2018}`.
- `EarlyStoppingEstimator` renamed `LowFidelityEstimator` (alias kept).

### Removed
- `ksplit`, `ModelSave`, `get_all_genes`, `get_search_space`, `tsmoothie` dependency.
