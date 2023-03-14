# NasLib
Neural architecture search lib in the making.  
Refactoring code from MPhys project.  
The code was originally written to build a CNN with integrated CBAM modules to classify cardiac arrhythmias (9-classes). 

<b>Search space</b>  
ResBlock-based with CBAM add-ons. This specific block structure is already proven. Hence, given the time constraints and computational limitations
this was my starting point.

<b>Search Strategy</b>  
Evolutionary. Currently 4 different sexual crossover operators, 1 mutation operartor which needs to be investigated further, 
1 asexual reproduction operator (for finer tuning). 

<b>Evaluation Strategy</b>  
Currently training model until plateau. This is expensive and, although works, requires a different approach.
Methods to try:
  - Bayesian optimisation
  - weight sharing
  - early stopping triggered by gradient acceleration (de). Statistical variance perhaps a problem here...
  - single layer optimisation (much faster)

<b>Moving forward...</b>
  - optimise current methods
  - equivalent 1D methods for corresponding datasets (e.g. no wavelet transform) and to increase speed
  - Minimal search spaces c.f. NEAT
  - optimise crossover operators
  - expand genetics
  - add speciation
  - add several more eval strategies: Bayesian Opt., Reinforcement Learning
  - add ViT and Transformer arch
   