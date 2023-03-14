# NasLib
Neural architecture search lib

Refactoring code from MPhys project on cardiac arrhythmia classification.

The code was originally written to build a CNN with integrated CBAM modules. 

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
  - early stopping triggered by gradient acceleration (de)
  - single layer optimisation (much much faster)
