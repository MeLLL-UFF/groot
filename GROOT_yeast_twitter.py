
import copy
import json
import logging
import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import time


from src.experiments.aux_code import *
from datasets.get_datasets import *
from src.brkga.genetic import genetic as brkga
from src.brkga_variation.genetic import genetic as brkga_var
from src.s_genetic.genetic import *
from src.predicate import *

logging.basicConfig(level=logging.INFO)
N_ROUNDS = 5
EXPERIMENT_NAME = 'yeast_twitter'

def get_k_best_individuals(population, k):
    best_ind = []
    all_fitness = []
    for i in range(0, len(population)):
        all_fitness.append((population[i].fitness.values[0], i))
    
    sorted_ind = sorted(all_fitness, reverse=False, key=lambda tup: tup[0])
    for i in range(0, k):
        best_ind.append(population[sorted_ind[i][1]])
    return best_ind

source = 'proteinclass'
target = 'accounttype'

kb = json.loads(open('src/experiments/kb.txt').readline())
kb_source = kb['yeast']
kb_target = kb['twitter']

pred_target = create_pred_target(kb_target)

yeast_dataset = datasets.load('yeast', kb_source, target=source, seed=441773, balanced=0) #facts, pos, neg
twitter_dataset = datasets.load('twitter', kb_target, target=target, seed=441773, balanced=0) #facts, pos, neg

ss = []
with open('src/experiments/structures.json', 'r') as f:
    ss = json.loads(f.readline())

src_struct = copy.deepcopy(ss[EXPERIMENT_NAME])
new_src_struct = []
for i in range(0, len(src_struct)):
    new_src_struct.append(define_individual(src_struct[i], i))  
structured_src = src_struct

train_pos, train_neg, train_facts = twitter_dataset[1], twitter_dataset[2], twitter_dataset[0]

logging.info("Iniciando algoritmo genético...")

logging.info("Iniciando otimização...")

mutation_rate_list = [round(x, 3) for x in list(np.arange(0.001, 0.01, 0.001)) + [0.01]]
crossover_rate_list = [round(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

space  = [Categorical([10, 30, 50], name='num_individuals'),
          Categorical(mutation_rate_list, name='mutation_rate'),
          Categorical(crossover_rate_list, name='crossover_rate')]

@use_named_args(space)
def objective(**params):
    res = genetic(new_src_struct, target, source, train_pos, train_neg, 
                      train_facts, kb_source, kb_target, pred_target,
                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                      mutation=params['mutation_rate'], crossover_type='tree_ind', revision='guided')

    return res[1][-1]


res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

logging.info("Best score=%.4f" % res_gp.fun)

logging.info(f"BEST RESULT: {res_gp.x}")

logging.info("Finalizada otimização!")

logging.info("Iniciando rounds...")

for _round in range(0, N_ROUNDS):
    print(f"ROUND {str(_round+1)}")
    res_s_genetic = genetic(new_src_struct, target, source, 
                    train_pos, train_neg, train_facts, 
                    kb_source, kb_target, pred_target,
                    NUM_GEN=14, pop_size=num_ind, 
                    mutation=mutation_rate, crossover=crossover_rate,
                    crossover_type='tree_ind', revision='guided')
    
    final_results = {}
    final_results[f'{source}->{target}'] = res_s_genetic
    
    individuals = get_k_best_individuals(res_s_genetic[0].population, 3)
    
    n_ind = 1
    for individual in individuals:
        rrefine = []
        rtransfer = []
        print("INDIVIDUO ", n_ind)
        refine, transfer = get_refine_transfer(individual, source, target, 'yeast', 'twitter')
        rrefine.append(refine)
        rtransfer.append(transfer)
        res = []
        inf = []
        for i in range(len(train_pos)):
            ttrain = []
            test_neg = []; test_pos = []; test_facts = []
            for index in range(0, len(train_pos)):
                if index == i:
                    ttrain = [train_pos[index], np.random.choice(train_neg[index], 2*len(train_pos[index])), train_facts[index]]
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
            res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
            res.append(res_)

            thisFile = f'boostsrl/test/results_{target}.db'
            base = os.path.splitext(thisFile)[0]
            os.rename(thisFile, base + ".txt")
            tt = open(f'boostsrl/test/results_{target}.txt', 'r').readlines()
            final = []
            for i in tt:
                final.append(i.replace('\n', ''))
            inf.append(final)


        final_results[f'test:{source}->{target}'] = res
        final_results[f'refine:{source}->{target}'] = rrefine
        final_results[f'transfer:{source}->{target}'] = rtransfer
        final_results[f'inf:{source}->{target}'] = inf
        save_groot_results(f'groot_experiments/s_genetic/{EXPERIMENT_NAME}_{source}_{target}_{str(crossover_rate)}_{str(mutation_rate)}_{str(num_ind)}_14', n_ind, final_results, source, target)
        n_ind += 1
        
logging.info("Finalizado algoritmo genético!")

logging.info("Iniciando BRKGA...")

logging.info("Iniciando otimização...")

num_elite_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.25, 0.05)) + [0.25]]
num_mutation_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.3, 0.05)) + [0.3]]
mutation_rate_list = [np.around(x, 3) for x in list(np.arange(0.001, 0.01, 0.001)) + [0.01]]
crossover_rate_list = [np.around(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

space  = [Categorical([10, 30, 50], name='num_individuals'),
          Categorical(mutation_rate_list, name='mutation_rate'),
          Categorical(crossover_rate_list, name='crossover_rate'),
          Categorical(num_elite_list, name='num_elite'),
          Categorical(num_mutation_list, name='num_mutation')]

@use_named_args(space)
def objective(**params):
    res = brkga(new_src_struct, target, source, train_pos, train_neg, 
                      train_facts, kb_source, kb_target, pred_target,
                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                      mutation=params['mutation_rate'], num_elite=params['num_elite'], 
                      num_mutation=params['num_mutation'])

    return res[1][-1]

res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

logging.info("Best score=%.4f" % res_gp.fun)

logging.info(f"BEST RESULT: {res_gp.x}")

logging.info("Finalizada otimização!")

logging.info("Iniciando rounds...")

for _round in range(0, n_rounds):
    print(f"ROUND {str(_round+1)}")
    res_brkga = brkga(new_src_struct, target, source, 
                    train_pos, train_neg, train_facts, 
                    kb_source, kb_target, pred_target,
                    NUM_GEN=14, pop_size=res_gp.x[0], 
                    mutation=res_gp.x[1], crossover=res_gp.x[2],
                    num_elite=res_gp.x[3], 
                    num_mutation=res_gp.x[4])
    
    final_results = {}
    final_results[f'{source}->{target}'] = res_brkga
    
    individuals = get_k_best_individuals(res_brkga[0].population, 3)
    
    n_ind = 1
    for individual in individuals:
        rrefine = []
        rtransfer = []
        print("INDIVIDUO ", n_ind)
        refine, transfer = get_refine_transfer(individual, source, target, 'yeast', 'twitter')
        rrefine.append(refine)
        rtransfer.append(transfer)
        res = []
        inf = []
        for i in range(len(train_pos)):
            ttrain = []
            test_neg = []; test_pos = []; test_facts = []
            for index in range(0, len(train_pos)):
                if index == i:
                    ttrain = [train_pos[index], np.random.choice(train_neg[index], 2*len(train_pos[index])), train_facts[index]]
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
            res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
            res.append(res_)

            thisFile = f'boostsrl/test/results_{target}.db'
            base = os.path.splitext(thisFile)[0]
            os.rename(thisFile, base + ".txt")
            tt = open(f'boostsrl/test/results_{target}.txt', 'r').readlines()
            final = []
            for i in tt:
                final.append(i.replace('\n', ''))
            inf.append(final)


        final_results[f'test:{source}->{target}'] = res
        final_results[f'refine:{source}->{target}'] = rrefine
        final_results[f'transfer:{source}->{target}'] = rtransfer
        final_results[f'inf:{source}->{target}'] = inf
        save_groot_results(f'groot_experiments/brkga/{EXPERIMENT_NAME}_{source}_{target}_{str(res_gp.x[2])}_{str(res_gp.x[1])}_{str(res_gp.x[0])}_14_{str(res_gp.x[3])}_{str(res_gp.x[4])}', n_ind, final_results, source, target)
        n_ind += 1


logging.info("Finalizado BRKGA!")

logging.info("Iniciando BRKGA variation...")

logging.info("Iniciando otimização...")

num_elite_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.25, 0.05)) + [0.25]]
num_mutation_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.3, 0.05)) + [0.3]]
mutation_rate_list = [np.around(x, 3) for x in list(np.arange(0.001, 0.01, 0.001)) + [0.01]]
crossover_rate_list = [np.around(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

space  = [Categorical([10, 30, 50], name='num_individuals'),
          Categorical(mutation_rate_list, name='mutation_rate'),
          Categorical(crossover_rate_list, name='crossover_rate'),
          Categorical(num_elite_list, name='num_elite'),
          Categorical(num_mutation_list, name='num_mutation')]

@use_named_args(space)
def objective(**params):
    res = brkga_var(new_src_struct, target, source, train_pos, train_neg, 
                      train_facts, kb_source, kb_target, pred_target,
                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                      mutation=params['mutation_rate'], num_top=params['num_elite'], 
                      num_bottom=params['num_mutation'], revision='guided')

    return res[1][-1]

res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

logging.info("Best score=%.4f" % res_gp.fun)

logging.info(f"BEST RESULT: {res_gp.x}")

logging.info("Finalizada otimização!")

logging.info("Iniciando rounds...")

for _round in range(0, n_rounds):
    print(f"ROUND {str(_round+1)}")
    res_brkga_var = brkga_var(new_src_struct, target, source, 
                    train_pos, train_neg, train_facts, 
                    kb_source, kb_target, pred_target,
                    NUM_GEN=14, pop_size=res_gp.x[0], 
                    mutation=res_gp.x[1], crossover=res_gp.x[2],
                    num_top=res_gp.x[3], 
                    num_bottom=res_gp.x[4], revision='guided')
    
    final_results = {}
    final_results[f'{source}->{target}'] = res_brkga_var
    
    individuals = get_k_best_individuals(res_brkga_var[0].population, 3)
    
    n_ind = 1
    for individual in individuals:
        rrefine = []
        rtransfer = []
        print("INDIVIDUO ", n_ind)
        refine, transfer = get_refine_transfer(individual, source, target, 'yeast', 'twitter')
        rrefine.append(refine)
        rtransfer.append(transfer)
        res = []
        inf = []
        for i in range(len(train_pos)):
            ttrain = []
            test_neg = []; test_pos = []; test_facts = []
            for index in range(0, len(train_pos)):
                if index == i:
                    ttrain = [train_pos[index], np.random.choice(train_neg[index], 2*len(train_pos[index])), train_facts[index]]
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
            res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
            res.append(res_)

            thisFile = f'boostsrl/test/results_{target}.db'
            base = os.path.splitext(thisFile)[0]
            os.rename(thisFile, base + ".txt")
            tt = open(f'boostsrl/test/results_{target}.txt', 'r').readlines()
            final = []
            for i in tt:
                final.append(i.replace('\n', ''))
            inf.append(final)


        final_results[f'test:{source}->{target}'] = res
        final_results[f'refine:{source}->{target}'] = rrefine
        final_results[f'transfer:{source}->{target}'] = rtransfer
        final_results[f'inf:{source}->{target}'] = inf
        save_groot_results(f'groot_experiments/brkga_var/{EXPERIMENT_NAME}_{source}_{target}_{str(res_gp.x[2])}_{str(res_gp.x[1])}_{str(res_gp.x[0])}_14_{str(res_gp.x[3])}_{str(res_gp.x[4])}', n_ind, final_results, source, target)
        n_ind += 1

logging.info("Finalizado BRKGA variation!")
