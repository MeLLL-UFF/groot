import json
import logging
import math
import multiprocessing
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
n_rounds = 10
num_processes = multiprocessing.cpu_count()
num_amounts = [0.1, 0.2, 0.4, 0.6, 0.8]

def get_k_best_individuals(population, k):
    best_ind = []
    all_fitness = []
    for i in range(0, len(population)):
        all_fitness.append((population[i].fitness.values[0], i))
    
    sorted_ind = sorted(all_fitness, reverse=False, key=lambda tup: tup[0])
    for i in range(0, k):
        best_ind.append(population[sorted_ind[i][1]])
    return best_ind

def get_train_neg_pos(source, target, source_pred, target_pred):
    train_neg = []
    train_pos = []
    
    with open(f'groot_experiments/{source}_{target}_{source_pred}_{target_pred}/neg_tmp.txt', 'r') as f:
        train_neg = json.loads(f.readline())
        
    with open(f'groot_experiments/{source}_{target}_{source_pred}_{target_pred}/pos_tmp.txt', 'r') as f:
        train_pos = json.loads(f.readline())
        
    with open(f'groot_experiments/{source}_{target}_{source_pred}_{target_pred}/facts_tmp.txt', 'r') as f:
        train_facts = json.loads(f.readline())
        
    return train_pos, train_neg, train_facts
    

experiments = {'experiment_1': ['pageclass', 'accounttype', 'webkb', 'twitter']}
# experiments = {'experiment_1': ['teamplayssport', 'companyeconomicsector', 'nell_sports', 'nell_finances']}
for experiment in experiments:
    logging.info(f"EXPERIMENT SOURCE {experiments[experiment][2]}_{experiments[experiment][0]}")
    logging.info(f"EXPERIMENT TARGET {experiments[experiment][3]}_{experiments[experiment][1]}")
    
    source_kb = experiments[experiment][2]
    target_kb = experiments[experiment][3]
    source_pred = experiments[experiment][0]
    target_pred = experiments[experiment][1]
    
    if target_kb in ['webkb', 'yeast', 'yago', 'yago2s', 'yeast2']:
        num_processes = int(multiprocessing.cpu_count()/2)
        
    source = source_pred
    target = target_pred

    bk = json.loads(open('src/experiments/kb.txt').readline())
    kb_source = bk[source_kb]
    kb_target = bk[target_kb]

    pred_target = create_pred_target(kb_target)
    
#     source_dataset = datasets.load(source_kb, kb_source, target=source, seed=441773, balanced=0) #facts, pos, neg
    target_dataset = datasets.load(target_kb, kb_target, target=target, seed=441773, balanced=0) #facts, pos, neg

    ss = []
    with open('src/experiments/structures.json', 'r') as f:
        ss = json.loads(f.readline())

    src_struct = copy.deepcopy(ss[f"{source_kb}_{source_pred}"])
    new_src_struct = []
    for i in range(0, len(src_struct)):
        new_src_struct.append(define_individual(src_struct[i], i))  
    structured_src = src_struct

    
#     train_pos, train_neg, train_facts = get_train_neg_pos(source_kb, target_kb, source, target)
    train_pos, train_neg, train_facts = target_dataset[1], target_dataset[2], target_dataset[0]

    logging.info("Iniciando algoritmo genético...")

#     logging.info("Iniciando otimização...")
    
    mutation_rate_list = [round(x, 2) for x in list(np.arange(0.1, 0.4, 0.05))]
    crossover_rate_list = [round(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

    space  = [Categorical([10, 30, 50], name='num_individuals'),
              Categorical(mutation_rate_list, name='mutation_rate'),
              Categorical(crossover_rate_list, name='crossover_rate')]

    for amount in num_amounts:
        logging.info(f"NUMBER OF FOLDS: {len(train_pos)}")
        for fold in range(0, len(train_pos)):
            num_examples_pos = math.floor(amount*len(train_pos[fold]))
            if num_examples_pos < 3: num_examples_pos = 3
            tmp_pos = train_pos[fold][:num_examples_pos]
            
            logging.info(f"FOLD: {fold}")
            best_res = []
            val_pos, val_neg, val_facts = get_train_test([tmp_pos], [train_neg[fold]], [train_facts[fold]],  n_folds=3)
            
            logging.info("Iniciando otimização...")
            
            for i in range(len(val_pos)):

                test = []
                ttrain = []
                test_pos = []
                test_neg = [] 
                test_facts = []
                for index in range(0, len(val_pos)):
                    if index == i:
                        ttrain = [val_pos[index], val_neg[index]]
                        test_facts.extend(val_facts[index])
                    else:
                        test_pos.extend(val_pos[index])
                        test_neg.extend(val_neg[index])
                        test_facts.extend(val_facts[index])
                test = [test_pos, test_neg, test_facts]
    #             ttrain.append(test[2])

                train_pos_gen = [ttrain[0], test[0]]
                train_neg_gen = [ttrain[1], test[1]]
                train_facts_gen = [test[2], test[2]]
                
               
                @use_named_args(space)
                def objective(**params):
                    res = genetic(new_src_struct, target, source, train_pos_gen, train_neg_gen, train_facts_gen,
                                      kb_source, kb_target, pred_target,
                                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                                      mutation=params['mutation_rate'], crossover_type='tree_ind', revision='guided',
                                      num_processes=num_processes)

                    return res[1][-1]
                res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

                logging.info("Best score=%.4f " % res_gp.fun)
                logging.info(f"FOLD {i} - BEST RESULT: {res_gp.x}")
                best_res.append((res_gp.fun, fold, res_gp.x))

            best_res = sorted(best_res, reverse=False, key=lambda tup: tup[0])
            logging.info(best_res)

            logging.info("Finalizada otimização!")

            logging.info("Iniciando rounds...")

            parameters = best_res[0][2]

            test = []
            ttrain = []
            test_pos = []
            test_neg = [] 
            test_facts = []
            for index in range(0, len(train_pos)):
                if index == fold:
                    ttrain = [tmp_pos, train_neg[index], train_facts[index]]
    #                 test_facts.extend(train_facts[index])
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
    #         ttrain.append(test[2])

            train_pos_gen = [ttrain[0], test[0]]
            train_neg_gen = [ttrain[1], test[1]]
            train_facts_gen = [ttrain[2], test[2]]
            
            logging.info(f"QTD POS: {len(train_pos_gen[0]), len(train_pos_gen[1])}")
            logging.info(f"QTD NEG: {len(train_neg_gen[0]), len(train_neg_gen[1])}")

            for _round in range(0, n_rounds):
                logging.info(f"ROUND {str(_round+1)}")
                logging.info(f"PARAMETERS: {parameters}, {fold}")

                res_s_genetic = genetic(new_src_struct, target_pred, source_pred, 
                                train_pos_gen, train_neg_gen, train_facts_gen, 
                                kb_source, kb_target, pred_target,
                                NUM_GEN=14, pop_size=parameters[0], 
                                mutation=parameters[1], crossover=parameters[2],
                                crossover_type='tree_ind', revision='guided',
                                num_processes=num_processes)


                final_results = {}
                final_results[f'{source_pred}->{target_pred}'] = res_s_genetic

                individuals = get_k_best_individuals(res_s_genetic[0].population, 3)

                n_ind = 1
                for individual in individuals:
                    rrefine = []
                    rtransfer = []
        #             logging.info("INDIVIDUO ", str(n_ind))
                    refine, transfer = get_refine_transfer(individual)
                    rrefine.append(refine)
                    rtransfer.append(transfer)
                    res = []
                    inf = []

                    res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
                    res.append(res_)

                    thisFile = f'boostsrl/test/results_{target_pred}.db'
                    base = os.path.splitext(thisFile)[0]
                    os.rename(thisFile, base + ".txt")
                    tt = open(f'boostsrl/test/results_{target_pred}.txt', 'r').readlines()
                    final = []
                    for i in tt:
                        final.append(i.replace('\n', ''))
                    inf.append(final)


                    final_results[f'test:{source}->{target}'] = res
                    final_results[f'refine:{source}->{target}'] = rrefine
                    final_results[f'transfer:{source}->{target}'] = rtransfer
                    final_results[f'inf:{source}->{target}'] = inf
                    save_groot_results(f'groot_experiments/{source_kb}_{target_kb}_{source_pred}_{target_pred}/s_genetic_tmp/{amount}_train_fold_{fold}/{str(parameters[2])}_{str(parameters[1])}_{str(parameters[0])}_15', n_ind, final_results, source_pred, target_pred, _round+1)
                    n_ind += 1

    logging.info("Finalizado algoritmo genético!")

    logging.info("Iniciando BRKGA...")

#     logging.info("Iniciando otimização...")
    
    num_elite_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.25, 0.05)) + [0.25]]
    num_mutation_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.3, 0.05)) + [0.3]]
    mutation_rate_list = [round(x, 2) for x in list(np.arange(0.1, 0.4, 0.05))]
    crossover_rate_list = [np.around(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

    space  = [Categorical([10, 30, 50], name='num_individuals'),
              Categorical(mutation_rate_list, name='mutation_rate'),
              Categorical(crossover_rate_list, name='crossover_rate'),
              Categorical(num_elite_list, name='num_elite'),
              Categorical(num_mutation_list, name='num_mutation')]
    
    for amount in num_amounts:
        logging.info(f"NUMBER OF FOLDS: {len(train_pos)}")
        for fold in range(0, len(train_pos)):
            num_examples_pos = math.floor(amount*len(train_pos[fold]))
            if num_examples_pos < 3: num_examples_pos = 3
            tmp_pos = train_pos[fold][:num_examples_pos]
            
            logging.info(f"FOLD: {fold}")
            best_res = []
            val_pos, val_neg, val_facts = get_train_test([tmp_pos], [train_neg[fold]], [train_facts[fold]],  n_folds=3)
            
            logging.info("Iniciando otimização...")
            
            for i in range(len(val_pos)):

                test = []
                ttrain = []
                test_pos = []
                test_neg = [] 
                test_facts = []
                for index in range(0, len(val_pos)):
                    if index == i:
                        ttrain = [val_pos[index], val_neg[index]]
                        test_facts.extend(val_facts[index])
                    else:
                        test_pos.extend(val_pos[index])
                        test_neg.extend(val_neg[index])
                        test_facts.extend(val_facts[index])
                test = [test_pos, test_neg, test_facts]
    #             ttrain.append(test[2])

                train_pos_gen = [ttrain[0], test[0]]
                train_neg_gen = [ttrain[1], test[1]]
                train_facts_gen = [test[2], test[2]]

                @use_named_args(space)
                def objective(**params):
                    res = genetic(new_src_struct, target, source, train_pos_gen, train_neg_gen, train_facts_gen,
                                      kb_source, kb_target, pred_target,
                                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                                      mutation=params['mutation_rate'], crossover_type='tree_ind', revision='guided',
                                      num_processes=num_processes)

                    return res[1][-1]
                res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

                logging.info("Best score=%.4f " % res_gp.fun)
                logging.info(f"FOLD {i} - BEST RESULT: {res_gp.x}")
                best_res.append((res_gp.fun, fold, res_gp.x))

            best_res = sorted(best_res, reverse=False, key=lambda tup: tup[0])
            logging.info(best_res)

            logging.info("Finalizada otimização!")

            logging.info("Iniciando rounds...")

            parameters = best_res[0][2]

            test = []
            ttrain = []
            test_pos = []
            test_neg = [] 
            test_facts = []
            for index in range(0, len(train_pos)):
                if index == fold:
                    ttrain = [tmp_pos, train_neg[index], train_facts[index]]
    #                 test_facts.extend(train_facts[index])
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
    #         ttrain.append(test[2])

            train_pos_gen = [ttrain[0], test[0]]
            train_neg_gen = [ttrain[1], test[1]]
            train_facts_gen = [ttrain[2], test[2]]
            
            logging.info(f"QTD POS: {len(train_pos_gen[0]), len(train_pos_gen[1])}")
            logging.info(f"QTD NEG: {len(train_neg_gen[0]), len(train_neg_gen[1])}")

            for _round in range(0, n_rounds):
                logging.info(f"ROUND {str(_round+1)}")


                res_brkga = brkga(new_src_struct, target_pred, source_pred, 
                                train_pos_gen, train_neg_gen, train_facts_gen, 
                                kb_source, kb_target, pred_target,
                                NUM_GEN=14, pop_size=parameters[0], 
                                mutation=parameters[1], crossover=parameters[2],
                                num_elite=parameters[3], 
                                num_mutation=parameters[4], 
                                num_processes=num_processes)


                final_results = {}
                final_results[f'{source_pred}->{target_pred}'] = res_brkga

                individuals = get_k_best_individuals(res_brkga[0].population, 3)

                n_ind = 1
                for individual in individuals:
                    rrefine = []
                    rtransfer = []
        #             logging.info("INDIVIDUO ", str(n_ind))
                    refine, transfer = get_refine_transfer(individual)
                    rrefine.append(refine)
                    rtransfer.append(transfer)
                    res = []
                    inf = []

                    res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
                    res.append(res_)

                    thisFile = f'boostsrl/test/results_{target_pred}.db'
                    base = os.path.splitext(thisFile)[0]
                    os.rename(thisFile, base + ".txt")
                    tt = open(f'boostsrl/test/results_{target_pred}.txt', 'r').readlines()
                    final = []
                    for i in tt:
                        final.append(i.replace('\n', ''))
                    inf.append(final)


                    final_results[f'test:{source}->{target}'] = res
                    final_results[f'refine:{source}->{target}'] = rrefine
                    final_results[f'transfer:{source}->{target}'] = rtransfer
                    final_results[f'inf:{source}->{target}'] = inf
                    save_groot_results(                                                                            f'groot_experiments/{source_kb}_{target_kb}_{source_pred}_{target_pred}/brkga_tmp/{amount}_train_fold_{fold}/{str(parameters[2])}_{str(parameters[1])}_{str(parameters[0])}_15_{str(parameters[3])}_{str(parameters[4])}', 
                                                                              n_ind, final_results, source_pred, target_pred, _round+1)
                    n_ind += 1

    logging.info("Finalizado BRKGA!")

    logging.info("Iniciando BRKGA variation...")

#     logging.info("Iniciando otimização...")
    
    num_elite_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.25, 0.05)) + [0.25]]
    num_mutation_list = [np.around(x, 2) for x in list(np.arange(0.1, 0.3, 0.05)) + [0.3]]
    mutation_rate_list = [round(x, 2) for x in list(np.arange(0.1, 0.4, 0.05))]
    crossover_rate_list = [np.around(x, 2) for x in list(np.arange(0.6, 0.95, 0.05)) + [0.95]]

    space  = [Categorical([10, 30, 50], name='num_individuals'),
              Categorical(mutation_rate_list, name='mutation_rate'),
              Categorical(crossover_rate_list, name='crossover_rate'),
              Categorical(num_elite_list, name='num_elite'),
              Categorical(num_mutation_list, name='num_mutation')]
    
    for amount in num_amounts:
        logging.info(f"NUMBER OF FOLDS: {len(train_pos)}")
        for fold in range(0, len(train_pos)):
            num_examples_pos = math.floor(amount*len(train_pos[fold]))
            if num_examples_pos < 3: num_examples_pos = 3
            tmp_pos = train_pos[fold][:num_examples_pos]
            
            logging.info(f"FOLD: {fold}")
            best_res = []
            val_pos, val_neg, val_facts = get_train_test([tmp_pos], [train_neg[fold]], [train_facts[fold]],  n_folds=3)
            
            logging.info("Iniciando otimização...")
            
            for i in range(len(val_pos)):

                test = []
                ttrain = []
                test_pos = []
                test_neg = [] 
                test_facts = []
                for index in range(0, len(val_pos)):
                    if index == i:
                        ttrain = [val_pos[index], val_neg[index]]
                        test_facts.extend(val_facts[index])
                    else:
                        test_pos.extend(val_pos[index])
                        test_neg.extend(val_neg[index])
                        test_facts.extend(val_facts[index])
                test = [test_pos, test_neg, test_facts]
    #             ttrain.append(test[2])

                train_pos_gen = [ttrain[0], test[0]]
                train_neg_gen = [ttrain[1], test[1]]
                train_facts_gen = [test[2], test[2]]

                @use_named_args(space)
                def objective(**params):
                    res = genetic(new_src_struct, target, source, train_pos_gen, train_neg_gen, train_facts_gen,
                                      kb_source, kb_target, pred_target,
                                      NUM_GEN=14, pop_size=params['num_individuals'], crossover=params['crossover_rate'],
                                      mutation=params['mutation_rate'], crossover_type='tree_ind', revision='guided',
                                      num_processes=num_processes)

                    return res[1][-1]
                res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

                logging.info("Best score=%.4f " % res_gp.fun)
                logging.info(f"FOLD {i} - BEST RESULT: {res_gp.x}")
                best_res.append((res_gp.fun, fold, res_gp.x))

            best_res = sorted(best_res, reverse=False, key=lambda tup: tup[0])
            logging.info(best_res)

            logging.info("Finalizada otimização!")

            logging.info("Iniciando rounds...")

            parameters = best_res[0][2]

            test = []
            ttrain = []
            test_pos = []
            test_neg = [] 
            test_facts = []
            for index in range(0, len(train_pos)):
                if index == fold:
                    ttrain = [tmp_pos, train_neg[index], train_facts[index]]
    #                 test_facts.extend(train_facts[index])
                else:
                    test_pos.extend(train_pos[index])
                    test_neg.extend(train_neg[index])
                    test_facts.extend(train_facts[index])
            test = [test_pos, test_neg, test_facts]
    #         ttrain.append(test[2])

            train_pos_gen = [ttrain[0], test[0]]
            train_neg_gen = [ttrain[1], test[1]]
            train_facts_gen = [ttrain[2], test[2]]
            
            logging.info(f"QTD POS: {len(train_pos_gen[0]), len(train_pos_gen[1])}")
            logging.info(f"QTD NEG: {len(train_neg_gen[0]), len(train_neg_gen[1])}")

            for _round in range(0, n_rounds):
                logging.info(f"ROUND {str(_round+1)}")
                print(f"PARAMETERS {parameters}")


                res_brkga_var = brkga_var(new_src_struct, target_pred, source_pred, 
                                train_pos_gen, train_neg_gen, train_facts_gen, 
                                kb_source, kb_target, pred_target,
                                NUM_GEN=14, pop_size=parameters[0], 
                                mutation=parameters[1], crossover=parameters[2],
                                num_top=parameters[3], 
                                num_bottom=parameters[4], revision='guided', num_processes=num_processes)


                final_results = {}
                final_results[f'{source_pred}->{target_pred}'] = res_brkga_var

                individuals = get_k_best_individuals(res_brkga_var[0].population, 3)

                n_ind = 1
                for individual in individuals:
                    rrefine = []
                    rtransfer = []
        #             logging.info("INDIVIDUO ", str(n_ind))
                    refine, transfer = get_refine_transfer(individual)
                    rrefine.append(refine)
                    rtransfer.append(transfer)
                    res = []
                    inf = []

                    res_ =  test_refine_transfer(kb_target, target, refine, transfer, ttrain, test)
                    res.append(res_)

                    thisFile = f'boostsrl/test/results_{target_pred}.db'
                    base = os.path.splitext(thisFile)[0]
                    os.rename(thisFile, base + ".txt")
                    tt = open(f'boostsrl/test/results_{target_pred}.txt', 'r').readlines()
                    final = []
                    for i in tt:
                        final.append(i.replace('\n', ''))
                    inf.append(final)


                    final_results[f'test:{source}->{target}'] = res
                    final_results[f'refine:{source}->{target}'] = rrefine
                    final_results[f'transfer:{source}->{target}'] = rtransfer
                    final_results[f'inf:{source}->{target}'] = inf
                    save_groot_results(f'groot_experiments/{source_kb}_{target_kb}_{source_pred}_{target_pred}/brkga_var_tmp/{amount}_train_fold_{fold}/{str(parameters[2])}_{str(parameters[1])}_{str(parameters[0])}_15_{str(parameters[3])}_{str(parameters[4])}', n_ind, final_results, source_pred, target_pred, _round+1)
                    n_ind += 1

    logging.info("Finalizado BRKGA VARIATION!")

