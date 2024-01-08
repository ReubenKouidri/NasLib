# NasLib Project

### Overview
NasLib is a refactored version of an MPhys project aimed at building
Convolutional Neural Networks (CNNs) with integrated Convolutional Block Attention Modules (CBAM)
for classifying cardiac arrhythmias across 9 classes. 

### Search Space
- <b>Cell-Based</b>: ResBlock with CBAM, a proven structure for this application.
- <b>Improvements to Consider</b>:
  - Minimize complexity, drawing inspiration from NEAT (NeuroEvolution of Augmenting Topologies).
  - Integrate Vision Transformers (ViTs).

### Search Strategy
- <b>Current Strategies</b>: evolutionary approach with 4 sexual crossover operators, 1 mutation operator, and 
    1 asexual reproduction operator (to search the local parameter space).
- <b>Future Modifications</b>: 
  - Speciation.
  - Random search as a baseline.
  
### Evaluation Strategy
- <b>Current Method</b>: early stopping after approximately 20 epochs, though this proves expensive and unreliable.
- <b>Proposed Methods</b>:
  - Implement early stopping triggered by gradient acceleration.
  - Single layer optimization for increased speed.
  - Sequential search (search for the best individual components independent of the rest).
  - Integration of speedy performance predictors to expedite the evaluation process.