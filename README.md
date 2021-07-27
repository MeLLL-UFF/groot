# GROOT - Genetic algorithms to aid  tRansfer Learning with bOOsTsrl

![GROOT][1.1]

This framework is made to offer transfer learning between relational domains, using genetic algorithms. 

Transfer learning with relational data is a challenge. The best solution to transfer is to find a mapping between the relations from the source and target datasets. GROOT finds the best mapping using genetic algorithm, Biased random-key genetic algorithm (BRKGA) [1] and a variation of BRKGA, built by us. The generated models are evaluated by RDN-Boost[2]. The variation divides the population into the top - with the best individuals -, the bottom, with the worst and middle, with individuals that are neither top nor bottom. 

This project is the result of a master's dissertation called <i>Transfer learning for boosted relational dependency networks through genetic algorithms</i>. The experiments aim to answer the following questions:

- Q1: Does GROOT perform better than learning from scratch?
- Q2: Does GROOT perform better than another transfer learning framework?
- Q3: Does GROOT reach good results in a viable time?

We compare our results with TreeBoostler framework [3], to answer Q2, and with RDN-Boost and RDN-B-1 (when using only one tree in RDN-Boost), to answer Q1. Both comparisons are used to answer Q3.

## Infrastructure

The algorithms are implemented in:

- src/s_genetic: the simple genetic algorithm
- src/brkga: BRKGA implementation
- src/brkga_variation: a modified BRKGA

In <i>src/experiments</i>, you can find:

- structures.json: the trees made by the source dataset using RDN-Boost
- kb.txt: knowledge base for each dataset used in the experiments
- aux_code.py: implementation to help when pre-processing data and processing results

The result for each experiment can be found in <i>groot_experiments</i>. Each folder name has the pattern

```
{source dataset}_{target dataset}_{from predicate}_{to predicate}
```

Each experiment ran 10 rounds, for each algorithm. 

## Installation

### Dependencies

GROOT requires:

- Python (>= 3)
- boostsrl 
- deap (>= 1.3.1)
- sklearn
- pandas (>= 1.2.4)
- scikit-optimize (if you desire to use the experiment.py file) (>= 0.8.1)

All the dependencies can be found on requirements.txt and could be installed using the command

```
pip install -r requirements.txt
```

## How to use

You could run the file experiment.py, defining what transfer you desire or relying on this file to build your experiment. In experiment.py, before the main experiment, we run an optimization of the hyperparameters to genetic algorithms.


## Contact

You can contact me for any problem in:

- E-mail: letfreirefigueiredo@gmail.com \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lfreire@cos.ufrj.br 

## References


[1] Gonçalves, José Fernando, and Mauricio GC Resende. "Biased random-key genetic algorithms for combinatorial optimization." Journal of Heuristics 17.5 (2011): 487-525.

[2] http://pages.cs.wisc.edu/~tushar/rdnboost/index.html

[3] https://github.com/MeLL-UFF/TreeBoostler


[1.1]:https://raw.githubusercontent.com/MeLL-UFF/groot/master/groot.png (groot icon)
