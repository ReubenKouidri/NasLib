# NasLib Project

### Overview
NasLib is an evolutionary neural architecture search (ENAS) framework. It provides a template for encoding
Modules as genes which can then be expressed and trained. Currently supports the following features:

### Search Space
- <b>Cell-Based</b>: ResBlock with CBAM, a proven structure for image classification (https://arxiv.org/abs/1807.06521).

### Search Strategy
- <b>Current Strategies</b>:
  - evolutionary approach with 4 sexual crossover operators, 1 mutation operator, and 
    1 asexual reproduction operator (to search the local parameter space).
  - Random Search: used as a baseline for benchmarking other methods (https://www.automl.org/wp-content/uploads/NAS/NAS_checklist.pdf).
  

  
### Evaluation Strategy
- <b>Early Stopping Estimator</b>: low-fidelity approach - used to benchmark other methods to follow.
