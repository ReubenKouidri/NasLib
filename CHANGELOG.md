# Changelog

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
