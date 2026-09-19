# NasLib / dnasty: review and roadmap (September 2026)

A five-years-on review of the project against the current NAS and ECG literature. Sections:

1. [Code structure, design and implementation](#1-code-structure-design-and-implementation)
2. [Datasets](#2-datasets)
3. [Search strategies](#3-search-strategies-five-families-that-work)
4. [Search space](#4-search-space-current-vs-literature)
5. [Packaging](#5-packaging)
6. [Measuring performance before and after a Mojo port](#6-measuring-performance-before-and-after-a-mojo-port)
7. [Roadmap](#7-roadmap)
8. [References](#8-references)

Status of this document: the fixes listed under 1.1 and the packaging skeleton in section 5 are done in
version 0.1.0. Everything else is the roadmap.

---

## 1. Code structure, design and implementation

### 1.1 Correctness bugs (fixed in 0.1.0)

| # | Where (pre-0.1.0 paths) | Problem | Fix |
|---|---|---|---|
| 1 | `search_space/cbam/genetics/genome.py` (`sync_genes`), `defaults/model.py` | The last layer was forced to `Softmax`, then `nn.CrossEntropyLoss` applied log-softmax again. Double softmax flattens gradients and caps the loss; every fitness value the search produced was affected. `nn.Softmax()` was also built without `dim`. | Output layer emits raw logits (no activation, no dropout). |
| 2 | `defaults/trainer.py` | Accuracy was `correct / len(dataloader)`, i.e. divided by the number of *batches*. Only correct because the validation loader used `batch_size=1`, which also made evaluation very slow. | Count samples; evaluation batch size comes from config. |
| 3 | `defaults/trainer.py`, `estimators.py`, `main.py` | `random_split` was unseeded and each `Trainer` made its own split. The estimator's trainer and the post-search trainer validated on different splits, so the fitness-vs-score correlation plot mixed training and validation records. | One seeded split in `DataModule`, shared by every trainer; global `seed` in config. |
| 4 | `defaults/model.py` | `ConvBlock2d(..., bn=True)`: no such argument, `S_2RB2D2()` raised `TypeError`. The `batch_norm` exon of `ConvBlock2dGene` was ignored by the component. | `ConvBlock2d` takes `batch_norm`; call fixed. |
| 5 | `search_space/cbam/components/components.py` | Spatial attention gate was `conv → Sigmoid → BatchNorm`, so the gate left [0, 1] (deviates from CBAM, Woo et al. 2018). All blocks were conv→act→BN; ResNet and CBAM use conv→BN→act. | Blocks are conv→BN→act; the gate ends with the sigmoid. |
| 6 | `datasets/CPSCDataset.py` | Labels were read positionally from `reference300.csv` (300 rows) and matched to `sorted(listdir)` of `test100` (100 files) by index rather than by recording id. Missing secondary labels became class 0 (= sinus rhythm). | Join on recording id; missing labels are `-1`. |
| 7 | `tests/datasets/test_cpsc_dataset.py` | Expected `float64` tensors and tensor targets; the code yielded `float32` and Python ints. The tests failed. | Items are tensors; expectations fixed. |
| 8 | `my_utils/ksplit.py` | Decorator called `split_dataset(dataset, ratio, k)` against a two-argument signature. | Removed. |
| 9 | `my_utils/my_utils.py` | `np.Inf` (removed in NumPy 2.0) in dead code. | Removed. |
| 10 | `search_space/__init__.py` | `get_all_genes` imported a package named `search_space` and read an `__all__` that did not exist; `get_search_space` imported an empty module. | Removed. |
| 11 | `my_utils/config.json`, tests | Absolute paths to another user's home directory; tests depended on the working directory and `sys.path`. | Packaged YAML configs, data root from `--data-dir` / `DNASTY_DATA_DIR`, pytest fixtures. |

Also: the README advertised four crossover operators, a mutation operator and an asexual operator. None were in the
tracked tree. They live in the git-ignored `moregacode/GeneticAlgoFinal.py`, which imports modules that no longer
exist (`the_creator`, `uga2_runs`, `ArrhythmiaDataset2D`), mixes `genome.genes` with `genome.chromosomes` and calls
`model.validate()` (not a torch method). Its operators are recorded in section 3 as the starting point for the port.

### 1.2 Design issues (roadmap)

- **Silent coercion.** `validate_feature` snaps invalid values to the nearest allowed one with a warning, and the
  genome test even relies on it (31→32, 50→64, 127→128). Constructors should raise; only `mutate` should clamp. The
  lookup is also O(len(range)) per call over `range(9, 10_001)`.
- **Metaprogramming in `GeneBase`.** `__getattr__`/`__setattr__` with a third positional `bypass` argument, a
  `__deepcopy__` that re-runs the constructor through `inspect.signature`, and `from_random` locating gene classes by
  module-path string. It works, but state such as `is_active` is lost on copy and none of it ports to Mojo. Replace
  with plain dataclass genes exposing `space`, `sample(rng)`, `mutate(rng)`, `to_module()` and structural
  `__eq__`/`__hash__`.
- **`Genome` has no `__eq__`/`__hash__`**, so duplicates can never be detected; the ordering dunders should go
  through `functools.total_ordering` or be dropped for explicit sort keys.
- **Validation mutates.** `is_genome_valid` calls `adjust_linear_genes`, which rewrites `out_features` during a check.
- **Hard-coded globals.** `num_classes`, `image_dims` and the input channel count were class constants; 0.1.0 makes them
  constructor arguments fed from config, but the chain space is still 2D-only.
- **Coupling.** Strategies import the `cbam` space directly and construct their own estimator; the old `my_utils`
  imported the top-level `datasets` package (which also collides with the HuggingFace `datasets` distribution). 0.1.0
  moves data into `dnasty.data` and injects the estimator; a `SearchSpace` protocol (sample, validate, encode,
  express) and an `Estimator` protocol are still missing.
- **No early stopping** in the "EarlyStoppingEstimator" (renamed `LowFidelityEstimator`), no results persistence
  before 0.1.0, no metric other than accuracy although CPSC and PTB-XL are scored with macro-F1 / macro-AUROC on
  heavily imbalanced classes.
- **Library hygiene.** `logging.basicConfig` at import time, star imports, heavy `skimage` import at package import,
  a Google-Drive default path in `StatisticsReporter`, `typing`/`twine`/`wheel`/`mypy` as runtime dependencies, no
  licence, an empty public API. Mostly addressed in 0.1.0.
- **Python-side performance.** The CWT (`pywt.cwt`, 63 scales × 1000 samples) plus `skimage.resize` runs for every
  record on every dataset construction with no cache; evaluation ran at batch size 1; every genome is deep-copied
  each generation. Training itself is inside torch kernels, which bounds what a Mojo port can gain (section 6).

## 2. Datasets

Current: 100 (+300) CPSC 2018 records, lead III only, first 4 s decimated by 2, min-max normalised, boxcar smoothed,
Mexican-hat CWT resized to 128×128; first label only; accuracy metric. MIT-BIH CSVs were tracked but unused.

| Dataset | Size | Why it matters here |
|---|---|---|
| **PTB-XL** v1.0.3 (PhysioNet; Wagner et al. 2020) | 21,837 12-lead 10 s records at 500/100 Hz, 71 statements, 18,885 patients | The de-facto benchmark. Predefined patient-stratified folds (1–8 train, 9 validation, 10 test) and published baselines (Strodthoff et al. 2021: 1D `xresnet1d` ≈ 0.93 macro-AUROC). WFDB format, ~1.7 GB at 100 Hz. Recommended primary dataset. |
| **CPSC 2018, full** (Liu et al. 2018) | 6,877 training / 2,954 hidden test records, 9 classes | The same task at 68× the data. Distributed inside the PhysioNet/CinC 2021 training set (WFDB); `dnasty data download cpsc2018` fetches it. |
| **PhysioNet/CinC 2021** (Reyna et al. 2021) | ~88k recordings from seven sources (CPSC, PTB-XL, G12EC, Chapman-Shaoxing, Ningbo, …) | Multi-source generalisation and the official challenge metric code. |
| **CODE-15%** (Ribeiro et al. 2021, Zenodo) | 345,779 exams, 233,770 patients, 6 classes, HDF5 | Large and open; already standardised to 4,096 samples. For final-architecture training, too big for search on a laptop. |
| **MIMIC-IV-ECG** (PhysioNet, credentialed) | ~800k 12-lead 10 s records | Largest available; final evaluation only. |
| MIT-BIH | 48 × 30 min, 2 leads, beat labels | A beat-level task, different from rhythm-level 12-lead classification. If kept, use the inter-patient DS1/DS2 split; intra-patient splits inflate results (systematic review, arXiv 2503.07276). |

Extraction and processing:

- Load with `wfdb` instead of tracked `.mat`/CSV files (done: `dnasty data download {mitbih,ptbxl,cpsc2018}`).
- **Preprocess once and cache**: fixed-length windows to one `float16`/`int16` `.npy` memmap (or HDF5/Zarr) per split
  plus a labels table; the CWT-image variant cached the same way, keyed by a hash of the preprocessing config.
  Parallelise the one-off CWT with `joblib`.
- Filter and resample with `scipy.signal` (band-pass 0.5–45 Hz, resample to 100 or 250 Hz); per-record z-score
  instead of min-max; drop the boxcar smoother.
- Make the 2D transform optional. PTB-XL benchmarks, CODE and the CinC winners use **1D CNNs on raw multi-lead
  signals**; wavelet images cost 128×128 floats per record and discard lead and phase information. Use all 12 leads
  (or a configurable subset).
- Multi-label targets (`BCEWithLogitsLoss`) scored with macro-F1/AUROC; class-weighted or focal loss for the
  single-label CPSC variant; `DataLoader(num_workers>0, pin_memory=True, persistent_workers=True)`.
- `scipy.signal.cwt` was removed in SciPy 1.15; keep `pywt` (or `ssqueezepy` for a fast GPU CWT).

## 3. Search strategies: five families that work

Context: a few hundred evaluations on one machine. The literature is blunt about the baseline: with a small space and
a low-fidelity evaluator, **random search is hard to beat** (Li & Talwalkar 2019; Yang et al. 2020). Every strategy
below must be compared with random search under fixed seeds, identical budgets and at least three repeats; build that
harness first.

| # | Strategy | Key references | How it works | Why it fits this project | Variants worth having |
|---|---|---|---|---|---|
| 1 | **Regularised (aging) evolution** | Real et al. 2019; NSGA-Net (Lu et al. 2019); LEMONADE (Elsken et al. 2019); CNN-GA (Sun et al. 2020) | Population as a queue: tournament-select a parent, mutate, evaluate, add the child, drop the *oldest* member (not the worst). Mutation-only in the original. | Direct successor of the existing GA; the strongest simple baseline on NAS-Bench-101/201; trivially parallel. | (a) Crossover: NSGA-Net's ablation shows crossover improves the Pareto front; use CNN-GA's one-point crossover at block boundaries for variable-length genomes plus uniform crossover for per-gene hyper-parameters. The old `mean` blend is an arithmetic / BLX-α crossover; keep it as one operator but replace the unbounded `exp(generation/alpha)` factor with a fixed α ∈ [0, 1]. (b) NSGA-II multi-objective (accuracy vs. params/FLOPs) instead of the hard `max_num_params` reject. (c) Lamarckian / network-morphism children that inherit parent weights (LEMONADE). |
| 2 | **Bayesian optimisation with neural predictors** | BANANAS (White et al. 2021a); NASBOWL (Ru et al. 2021); BONAS; GRAF neural graph features (Kadlecová et al. 2024); White et al. 2021b | Fit a surrogate (MLP ensemble or GP with a Weisfeiler-Lehman graph kernel) on evaluated architectures; choose the next candidates by an acquisition function optimised with mutations of the best. | Best sample-efficiency per trained model at 100–500 evaluations, which is the budget available. | (a) Predictor-assisted evolution: score mutation children with the surrogate and train only the top-k (NPENAS). (b) Encoding: path encoding for cell DAGs, plain ordinal vectors for the current chain space. |
| 3 | **Zero-cost proxies / training-free ranking** | NASWOT (Mellor et al. 2021); Synflow / Zero-Cost-PT (Abdelfattah et al. 2021); ZiCo (Li et al. 2023); NAS-Bench-Suite-Zero (Krishnakumar et al. 2022); NEAR (2024) | Score an *untrained* network from one batch: activation-pattern diversity, synaptic flow, gradient mean/variance. | Replaces or filters before the low-fidelity estimator at ~1000× lower cost. Suite-Zero shows proxies are complementary and adding them to a surrogate improves prediction by up to 42 %; ZiCo is the only single proxy that consistently beats parameter count. | (a) Proxy-guided evolution: proxies as a cheap first-stage filter. (b) Multi-fidelity: successive halving / Hyperband over epochs, or learning-curve extrapolation (NAS-Bench-x11); this is what the low-fidelity estimator should become. |
| 4 | **One-shot / weight-sharing** | DARTS (Liu et al. 2019); PC-DARTS, DARTS-PT, β-DARTS; SPOS (Guo et al. 2020); Once-for-All (Cai et al. 2020) | Train one supernet containing every op; relax to continuous mixing weights (DARTS) or sample paths uniformly (SPOS) and then search sub-networks with an EA. | Highest sample-efficiency; SPOS + evolutionary search reuses the GA machinery and avoids DARTS' skip-connect collapse. Needs a fixed macro-skeleton with shape-preserving ops, which is a reason to redesign the space (section 4). | (a) SPOS uniform-sampling supernet + the EA (recommended). (b) DARTS-PT perturbation-based selection if a differentiable variant is wanted. |
| 5 | **LLM-guided evolution** | EvoPrompting (Chen et al. 2023); LLMatic (Nasir et al. 2024); GENIUS; RZ-NAS; GraphIR (2026) | A language model acts as the mutation/crossover operator over a textual or IR description of the architecture, optionally with a quality-diversity archive (MAP-Elites) instead of a single fitness. | Genes already compile to modules, so a JSON/IR genome is a natural prompt state. Strong results at tiny budgets; the cost is API calls rather than GPU-hours. | (a) LLM proposes, zero-cost proxy filters, trainer evaluates. (b) QD archive keyed on (params, depth) to keep diversity. |

Not recommended as a sixth family: reinforcement-learning controllers (NASNet, ENAS). Historically important,
superseded on sample-efficiency by 1–4.

Operator set to implement for the chain/cell genome (from the references above):

- **Mutation**: hyper-parameter step (±1 index in the allowed list), op replacement, block insert/delete for
  variable length, identity with probability p. Aging evolution applies exactly one mutation per child.
- **Crossover**: one-point at block boundaries (CNN-GA), uniform per gene, arithmetic/BLX-α for integer
  hyper-parameters (the old `mean` operator with a fixed α), plus *no crossover* as a control.
- **Selection**: tournament (k = 3–10 % of the population), aging removal, elitism ≤ 10 %.

What the old `moregacode` GA did, for reference: `mean` crossover blended every gene of two parents with a factor
`exp(generation/100)/2`; `random` crossover blended a random subset of genes; mutation added a bounded random integer
with rejection sampling; the top-k elites were cloned and mutated ("asexual"); children were validated by feature-map
size only.

## 4. Search space: current vs. literature

Current (`configs/*.yaml`, `create_gene_sequence`): `[ConvBlock2d × (1..2) → MaxPool2d → CBAM] × (1..2) → Flatten →
LinearBlock × 2`, one input channel, 128×128, *valid* (unpadded) convolutions with k ∈ [2, 16], channels in
{1, …, 128} as powers of two, pool k ∈ {2, 3, 4}, ReLU only, ≤ 1 M parameters, final map 5 < d < 25, linear width up to
10 k. Roughly 10⁶–10⁷ distinct genomes before validity filtering; most are rejected or dominated by the linear head.

Findings:

- **The classifier head dominates.** `flatten(25×25×128) = 80k` inputs × up to 10k neurons is where the parameter
  budget goes, which is why `adjust_linear_genes` exists. Global average pooling (standard since ResNet/SENet) removes
  the head problem, the `outdims` constraints and the `adjust_linear_genes` machinery in one move.
- **No padding + large kernels** shrink the map and couple depth to kernel size; the literature uses `same` padding
  with strided convolutions or pooling as explicit downsampling stages.
- **Missing structural choices**: skip connections beyond CBAM's residual, stride, dilation, depth-wise separable or
  grouped convolutions, per-stage kernel size, width multiplier, normalisation choice, dropout rate, activation
  (fixed ReLU). There is no DAG/cell topology, so none of the benchmark spaces (NAS-Bench-101/201, DARTS) or the
  one-shot methods apply.
- **ECG-specific ops absent**: 1D convolutions with large receptive fields, dilated/TCN blocks, Inception-1D
  multi-scale blocks, SE/CBAM on 1D, light attention/transformer blocks.
- **Constraint handling**: hard rejection plus in-place repair biases sampling toward small heads; the literature
  uses a parameter/FLOP objective (NSGA-II) or a soft penalty.

Recommendation: a **macro-skeleton with searchable cells**, dimension-agnostic (`nn.Conv1d` / `nn.Conv2d` chosen by
`dim`): `stem → S stages × (N cells, downsample between stages) → GAP → linear(num_classes)`. A cell is a small DAG
(NAS-Bench-201 style, 4 nodes) or a fixed-topology block with searchable op / kernel / width / attention. Op set:
`skip`, `conv k ∈ {3, 5, 7, 9, 15, 31}`, `dw-sep conv`, `dilated conv d ∈ {2, 4}`, `avg/max pool`, `SE`/`CBAM`, `none`.
Presets `tiny` (≈ today's budget), `small`, `medium`. Parameters and FLOPs computed analytically and exposed as a
second objective.

## 5. Packaging

Done in 0.1.0:

1. Distribution and import name `dnasty` (`naslib` collides with automl/NASLib). Check PyPI availability before the
   first upload.
2. `src/` layout; `datasets/` → `dnasty.data`; `my_utils` → `dnasty.utils`; explicit `__init__.py` in every
   sub-package; public API in `dnasty/__init__.py`.
3. `pyproject.toml` (PEP 621, hatchling): `requires-python >=3.10,<3.14`, lower-bound dependencies, extras `data`,
   `viz`, `dev`. On Intel macOS the resolver pins `torch<2.3` and `numpy<2` (the last builds with wheels for that
   platform).
4. `uv` workflow: `.python-version`, `uv.lock`, `uv sync --extra dev`, `uv build`.
5. Packaged YAML configs, data root via `--data-dir` / `DNASTY_DATA_DIR`, `dnasty` CLI (`search`, `data download`).
6. Data untracked (`git rm --cached`), `.gitignore` for `data/`, `datasets/`, `*.mat`, `*.pdf`.
7. Pytest with fixtures; data-dependent tests marked `data` and skipped without files.
8. `ruff`, `mypy` config, `pre-commit`, GitHub Actions matrix (3.10–3.13 × ubuntu/macos).
9. `LICENSE` (MIT), `CHANGELOG.md`, `__version__` from package metadata.

Still to do: publish to TestPyPI then PyPI with trusted publishing; optional `mkdocs-material` API docs; optionally
purge the historical data blobs with `git filter-repo` (rewrites history, needs a force-push, so a deliberate step).

## 6. Measuring performance before and after a Mojo port

Mojo status (September 2026): Mojo 1.0 beta shipped with Modular 26.3; the language was open-sourced under Apache 2.0
on 18 August 2026; it installs with `pip install mojo`, can be called from Python, and MAX supports PyTorch custom ops
written in Mojo. Mojo sits *beside* PyTorch (kernels and ops), not in place of it, so time inside torch kernels will not
change. The port targets Python-side hot paths and the benchmark has to isolate them.

1. **Profile first** on the refactored Python code with a fixed seed: `py-spy record` or `pyinstrument` on a full search
   with the real trainer and with the mock estimator. Record the fraction *f* of wall time outside torch kernels;
   Amdahl's law bounds the whole-run speed-up at 1 / (1 − f). Expected candidates: genome sampling, validation,
   mutation and crossover, `deepcopy`, zero-cost proxies, CWT and resampling, population bookkeeping.
2. **Benchmark suite** in `benchmarks/` with `pytest-benchmark` (JSON output, `--benchmark-compare`), one case per hot
   path with identical inputs from a seeded generator: `sample_population(1000)`, `mutate/crossover(10k)`,
   `validate(10k)`, `preprocess_record(1000 ECGs)`, `cwt_image(100)`, `zero_cost_proxy(50 nets)`, plus one
   end-to-end `search(tiny, mock estimator, 5 generations)`. `hyperfine` for CLI-level wall time.
3. **Metrics**: median and IQR of wall time over ≥ 10 runs after warm-up, CPU time, peak RSS (`/usr/bin/time -l` on
   macOS, `memray` for Python), throughput (genomes/s, records/s), and **parity**: with a shared PRNG (e.g. PCG32
   implemented in both languages) the outputs must be byte-identical for the same seed; otherwise compare
   distributions (KS test). The Python implementation stays the oracle in the Mojo tests.
4. **Fix the environment**: record machine, OS, Python, torch and Mojo versions in the results file; mains power,
   no other load.
5. **Freeze the baseline** as `benchmarks/results/<date>-<git-sha>.json` before any Mojo code, then report speed-ups
   with confidence intervals per function and end-to-end via `pytest-benchmark compare`.
6. **Port order**: pure-Python numeric loops first (validation, operators, proxies), then preprocessing, then
   optionally a custom CBAM op via MAX; measure after each step.

## 7. Roadmap

1. **Benchmark harness** (done: `dnasty benchmark`, `dnasty.benchmark`, `SyntheticEstimator`, `CachedEstimator`):
   seeded runs, fixed evaluation budget, random-search baseline, ≥ 3 repeats, results to `runs/`. Everything below is
   judged with it. First result on the 100-record CPSC subset with the 1-epoch estimator (40 evaluations, 3 seeds):
   random 0.317 ± 0.012 vs regularised evolution 0.308 ± 0.012, majority-class baseline 0.275, repeat noise of a
   single genome ≈ 0.04. The estimator is noise at this data size, which makes item 4 the prerequisite for any
   strategy comparison on real data.
2. **Genes without metaprogramming**: dataclass genes, strict validation, `Genome.__eq__`/`__hash__`, pure
   validity checks.
3. **Search space v2** (section 4): dimension-agnostic cells, GAP head, analytic params/FLOPs, `tiny/small/medium`.
4. **Data v2** (section 2): PTB-XL 1D loader with cached preprocessing, all leads, multi-label metrics (done:
   `PTBXLDataset` with predefined folds, six tasks and a float16 memmap cache; `Trainer` with `BCEWithLogitsLoss`
   and macro-AUROC / macro-F1; `ECGNet1d` reference; subset downloads). First result: ECGNet1d (104k params) on a 2,700-record subset reaches
   0.910 macro-AUROC with a repeat sd of 0.001 (CPSC 1-epoch estimator: 0.04), i.e. a usable fitness signal.
   CPSC full still to do. Searching PTB-XL
   needs item 3, since the `cbam` space is 2D.
5. **Strategies** (section 3): `RegularizedEvolution` with the operator library (done: mutation-only aging evolution,
   hyper-parameter / insert-conv / delete-conv mutations, one-point crossover behind `evolution.crossover_prob`;
   ablations still to run on real data), `NSGA2`,
   zero-cost-proxy and successive-halving estimators, BANANAS-style predictor; SPOS supernet once space v2 exists;
   LLM-guided operators as an experiment.
6. **Mojo**: profile, freeze the baseline, port the measured hot paths (section 6).

## 8. References

- Abdelfattah, M. S. et al. (2021). Zero-Cost Proxies for Lightweight NAS. ICLR.
- Cai, H. et al. (2020). Once-for-All: Train One Network and Specialize it for Efficient Deployment. ICLR.
- Chen, A. et al. (2023). EvoPrompting: Language Models for Code-Level Neural Architecture Search. NeurIPS.
- Elsken, T., Metzen, J. H., Hutter, F. (2019). Efficient Multi-objective NAS via Lamarckian Evolution (LEMONADE). ICLR.
- Guo, Z. et al. (2020). Single Path One-Shot NAS with Uniform Sampling (SPOS). ECCV.
- Kadlecová, G. et al. (2024). Surprisingly Strong Performance Prediction with Neural Graph Features (GRAF). ICML.
- Krishnakumar, A. et al. (2022). NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies. NeurIPS D&B.
- Li, G. et al. (2023). ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients. ICLR.
- Li, L., Talwalkar, A. (2019). Random Search and Reproducibility for NAS. UAI.
- Liu, F. et al. (2018). An Open Access Database for Evaluating the Algorithms of ECG Rhythm and Morphology Abnormality Detection (CPSC 2018). J. Med. Imaging Health Inform.
- Liu, H., Simonyan, K., Yang, Y. (2019). DARTS: Differentiable Architecture Search. ICLR.
- Lu, Z. et al. (2019). NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm. GECCO.
- Mellor, J. et al. (2021). Neural Architecture Search without Training (NASWOT). ICML.
- Nasir, M. U. et al. (2024). LLMatic: NAS via Large Language Models and Quality Diversity Optimization. GECCO.
- Real, E. et al. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI.
- Reyna, M. A. et al. (2021). Will Two Do? Varying Dimensions in Electrocardiography: PhysioNet/CinC Challenge 2021.
- Ribeiro, A. H. et al. (2020/2021). Automatic diagnosis of the 12-lead ECG using a deep neural network; CODE-15% dataset (Zenodo 4916206).
- Ru, B. et al. (2021). Interpretable NAS via Bayesian Optimisation with Weisfeiler-Lehman Kernels (NASBOWL). ICLR.
- Strodthoff, N. et al. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. IEEE JBHI.
- Sun, Y. et al. (2020). Automatically Designing CNN Architectures Using the Genetic Algorithm for Image Classification (CNN-GA). IEEE Trans. Cybernetics.
- Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.
- White, C. et al. (2021a). BANANAS: Bayesian Optimization with Neural Architectures for NAS. AAAI.
- White, C. et al. (2021b). How Powerful are Performance Predictors in Neural Architecture Search? NeurIPS.
- Woo, S. et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
- Yang, A., Esperança, P. M., Carlucci, F. M. (2020). NAS evaluation is frustratingly hard. ICLR.
- Systematic review of ECG arrhythmia classification standards and fair evaluation (2025). arXiv:2503.07276.
- Modular (2026). Modular 26.3: Mojo 1.0 Beta; Mojo open-sourced under Apache 2.0 (August 2026). modular.com/blog.
