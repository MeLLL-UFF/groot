####
# Auxiliary source
####
from boostsrl import boostsrl
import copy
import json
import re
import os
import random
import numpy as np
import sys
from sklearn.model_selection import KFold, train_test_split

def get_train_division(dataset):
    train_facts_source = []
    train_pos_source = []
    train_neg_source = []
    for i in range(0, len(dataset[0])):
        train_facts_source.extend(dataset[0][i])
        train_pos_source.extend(dataset[1][i])
        train_neg_source.extend(dataset[2][i])
    return train_facts_source, train_pos_source, train_neg_source


def split_folds(dataset, n_folds):
    # random.shuffle(dataset)
    res = []
    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(dataset):
        res.append(np.array(dataset)[test_index])
    return res


def split_train_test(dataset, test_size=0.3):
    X_train, X_test = train_test_split(dataset, test_size=test_size, random_state=42)
    return X_train, X_test

def get_train_test(dataset, i):
    # test_index = random.randint(0, len(dataset[0])-1)
    test_index = i
    print(test_index)
    if len(dataset[0]) >= 3:
        test_facts = dataset[0][test_index]
        test_pos = dataset[1][test_index]
        test_neg = dataset[2][test_index]

        dataset[0].remove(dataset[0][test_index])
        dataset[1].remove(dataset[1][test_index])
        dataset[2].remove(dataset[2][test_index])

        train_facts = dataset[0]
        train_pos = dataset[1]
        train_neg = dataset[2]

        # test_facts = dataset[0][0]
        # test_pos = dataset[1][0]
        # test_neg = dataset[2][0]
    elif len(dataset[0]) == 2:
        test_facts = dataset[0][test_index]
        test_pos = dataset[1][test_index]
        test_neg = dataset[2][test_index]

        dataset[0].remove(dataset[0][test_index])
        dataset[1].remove(dataset[1][test_index])
        dataset[2].remove(dataset[2][test_index])

        train_facts = split_folds(dataset[0][0], 2)
        train_pos = split_folds(dataset[1][0], 2)
        train_neg = split_folds(dataset[2][0], 2)
    else:
        train_facts, test_facts = split_train_test(dataset[0][0])
        train_pos, test_pos = split_train_test(dataset[1][0])
        train_neg, test_neg = split_train_test(dataset[2][0])

        train_facts = split_folds(train_facts, 2)
        train_pos = split_folds(train_pos, 2)
        train_neg = split_folds(train_neg, 2)
    
    return train_facts, train_pos, train_neg, test_facts, test_pos, test_neg


def get_branch(curr_value, next_value):
    if curr_value == '': 
        return next_value
    return '{},{}'.format(curr_value, next_value)


def define_individual(structured_tree, tree_number):
    individual_tree = []
    forceLearning=False
    target = structured_tree[0]
    nodes = structured_tree[1]
    for values, node in nodes.items():
        if values == '': 
            branch = '{} :- {}.'.format(target, node)
        else: branch = '{}.'.format(node)
        left_branch = 'true' if get_branch(values, 'true') in nodes or forceLearning else 'false'
        right_branch = 'true' if get_branch(values, 'false') in nodes or forceLearning else 'false'
        individual_tree.append('{};{};{};{};{}'.format(tree_number, values, 
                                                  branch, left_branch, right_branch))
    return individual_tree
                          

def create_structured_trees(model):
    structured_src = []
    for i in range(0, 10):
        try:
            structured_src.append(model.get_structured_tree(treenumber=i+1).copy())
        except:
            structured_src.append(model.get_structured_tree(treenumber='combine').copy())
           
    src_struct = copy.deepcopy(structured_src)
    # print(src_struct)
    new_src_struct = []
    for i in range(0, len(src_struct)):
        new_src_struct.append(define_individual(src_struct[i], i))  
    
    return structured_src, new_src_struct


def create_pred_target(kb):
    #pred_target é : pred_target = [('movie', '+,-'), ('director', '+'),...]
    pred_target = []
    for pred in kb:
        modes = ','.join([pred[occur.start()] for occur in re.finditer('[+\-]', pred)])
        pred_target.append((pred.split('(')[0], modes))
    return pred_target


def get_best_individual(population):
    best_result = population[0].fitness.values[0]
    ind = population[0]
    for i in population:
        if i.fitness.values[0] < best_result:
            best_result = i.fitness.values[0]
            ind = i
    return ind
    

def get_refine_transfer(ind, source, target, kb_source, kb_target):

    if not os.path.exists(f'src/experiments/{kb_source}_{kb_target}'):
            os.makedirs(f'src/experiments/{kb_source}_{kb_target}')

    refine = []
    for tree in ind.modified_src_tree:
        refine.extend(tree)
    transfer = ind.transfer.transfer
    with open(f'src/experiments/{kb_source}_{kb_target}/refine_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(refine))
        f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/transfer_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(transfer))
        f.close()
    return refine, transfer


def test_refine_transfer(kb, target, refine, transfer, train_dataset, test_dataset):
    background_train = boostsrl.modes(kb, [target], useStdLogicVariables=False, 
                                          maxTreeDepth=3, nodeSize=2, numOfClauses=8)
    model_tr = boostsrl.train(background_train, train_dataset[0], train_dataset[1], 
                                      train_dataset[2], refine=refine, transfer=transfer,
                                      trees=10)
    test_model = boostsrl.test(model_tr, test_dataset[0], test_dataset[1], 
                                       test_dataset[2], trees=10)
    return test_model.summarize_results(), test_model


def test_tree_b(source, target, kb_source, kb_target, transferred_structured, train_dataset, test_dataset, _transfer=None):
    sys.path.insert(0, '../TreeBoostler')

    from revision import revision
    from transfer import transfer
#     from mapping import *
    from boostsrl import boostsrl
    import os

    predicate = source
    to_predicate = target

    def print_function(message):
        global experiment_title
        global nbr
        experiment_title = f'Twitter->Yeast2: {source}->{target}'
        nbr = 1
        if not os.path.exists('experiments/' + experiment_title):
            os.makedirs('experiments/' + experiment_title)
        with open('experiments/' + experiment_title + '/' + str(nbr) + '_' + experiment_title + '.txt', 'a') as f:
            print(message, file=f)
            print(message)
    if _transfer:
        tr_file = _transfer
    else:
        tr_file = transfer.get_transfer_file(kb_source, kb_target, predicate, to_predicate, searchArgPermutation=True, allowSameTargetMap=False)
    new_target = to_predicate

    # transfer and revision theory
    # experiment_title = f'IMDB->cora: {source}->{target}'
    # nbr = 1
    background = boostsrl.modes(kb_target, [to_predicate], useStdLogicVariables=False, maxTreeDepth=3, nodeSize=2, numOfClauses=8)
    return revision.theory_revision(background, boostsrl, target, train_dataset[0], train_dataset[1], train_dataset[2], test_dataset[0], test_dataset[1], test_dataset[2], transferred_structured, transfer=tr_file, trees=10, max_revision_iterations=10, print_function=print_function)


def save_results(final_results, source, target, kb_source, kb_target):
    if not os.path.exists(f'src/experiments/{kb_source}_{kb_target}'):
            os.makedirs(f'src/experiments/{kb_source}_{kb_target}')

    with open(f'src/experiments/{kb_source}_{kb_target}/groot_train_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'{source}->{target}']:
            f.write(json.dumps(i[1]))
            f.write('\n')
            f.write(json.dumps(i[2]))
        f.close()
    # with open(f'src/experiments/{kb_source}_{kb_target}/groot_test_{source}_{target}.txt', 'w') as f:
    #     for i in final_results[f'groot_test:{source}->{target}']:
    #         f.write(json.dumps(i))
    #         f.write('\n')
    #     f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/groot_inf_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'inf_res:{source}->{target}']:
            f.write(json.dumps(i))
            f.write('\n')
        f.close()
    # with open(f'src/experiments/{kb_source}_{kb_target}/treeb_test_{source}_{target}.txt', 'w') as f:
    #     f.write(json.dumps(final_results[f'tree_test:{source}->{target}'][1]))
    #     f.write('\n')
    #     f.write(json.dumps(final_results[f'tree_test:{source}->{target}'][2]))
    #     f.write('\n')
    #     f.write(json.dumps(final_results[f'tree_test:{source}->{target}'][3]))
    #     f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/groot_train_brkga_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'brkga_{source}->{target}']:
            f.write(json.dumps(i[1]))
            f.write('\n')
            f.write(json.dumps(i[2]))
        f.close()
    # with open(f'src/experiments/{kb_source}_{kb_target}/groot_test_brkga_{source}_{target}.txt', 'w') as f:
    #     for i in final_results[f'groot_test_brkga:{source}->{target}']:
    #         f.write(json.dumps(i))
    #         f.write('\n')
    #     f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/groot_inf_brkga_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'inf_res_brkga:{source}->{target}']:
            f.write(json.dumps(i))
            f.write('\n')
        f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/groot_train_brkga_var_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'brkga_var_{source}->{target}']:
            f.write(json.dumps(i[1]))
            f.write('\n')
            f.write(json.dumps(i[2]))
        f.close()
    # with open(f'src/experiments/{kb_source}_{kb_target}/groot_test_brkga_var_{source}_{target}.txt', 'w') as f:
    #     for i in final_results[f'groot_test_brkga_var:{source}->{target}']:
    #         f.write(json.dumps(i))
    #         f.write('\n')
    #     f.close()
    with open(f'src/experiments/{kb_source}_{kb_target}/groot_inf_brkga_var_{source}_{target}.txt', 'w') as f:
        for i in final_results[f'inf_res_brkga_var:{source}->{target}']:
            f.write(json.dumps(i))
            f.write('\n')
        f.close()
