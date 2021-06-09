####
# Auxiliary source
####
from boostsrl import boostsrl
import copy
import json
import math
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

def get_train_test(pos, neg, facts,  n_folds=3):
    if len(pos) > 0:
        pos = [x for y in pos for x in y]
        neg = [x for y in neg for x in y]
        facts =  [x for y in facts for x in y]

    pos_ex = []
    neg_ex = []
    facts_ex = [facts]*n_folds

    #calculate positives
    amount = math.ceil(len(pos)/n_folds)
    for i in range(n_folds):
        pos_ex.append(pos[:amount])
        pos = pos[amount:]

    #calculate negatives
    amount = math.ceil(len(neg)/n_folds)
    for i in range(n_folds):
        neg_ex.append(neg[:amount])
        neg = neg[amount:]

    return pos_ex, neg_ex, facts_ex


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
        modes = ','.join([pred[occur.start()] for occur in re.finditer('[+\-\`]', pred)])
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
    

def get_refine_transfer(ind):
    refine = []
    for tree in ind.modified_src_tree:
        refine.extend(tree)
    transfer = ind.transfer.transfer
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


def save_examples(pos, neg, path):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    with open(f'{path}/pos.txt', 'w') as f:
        f.write(json.dumps(pos))
    f.close()

    with open(f'{path}/neg.txt', 'w') as f:
        f.write(json.dumps(neg))
    f.close()

def save_base_results(rdn_b_result, rdn_result, tree_b_result, path):
    """ path is like genetic_type/experiment_name_genetic_info
    """
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    with open(f'{path}/rdn_boost.txt', 'w') as f:
        for i in rdn_b_result:
                f.write(json.dumps(i[1]))
                f.write('\n')
                f.write(json.dumps(i[2]))
                f.write('\n')
                f.write(json.dumps(i[3]))
                f.write('\n')
                f.write(json.dumps(i[4]))
                f.write('\n')

        f.close()

    with open(f'{path}/rdn.txt', 'w') as f:
        for i in rdn_result:
                f.write(json.dumps(i[1]))
                f.write('\n')
                f.write(json.dumps(i[2]))
                f.write('\n')
                f.write(json.dumps(i[3]))
                f.write('\n')
                f.write(json.dumps(i[4]))
                f.write('\n')

        f.close()

    with open(f'{path}/treeb.txt', 'w') as f:
        for i in tree_b_result:
                f.write(json.dumps(i[1]))
                f.write('\n')
                f.write(json.dumps(i[2]))
                f.write('\n')
                f.write(json.dumps(i[3]))
                f.write('\n')

        f.close()


def save_groot_results(path, individual_number, final_results, source, target):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    rounds = [filename.split('_')[1] for filename in os.listdir(f'{path}') if filename.startswith("round_")]

    last_folder = 1
    if len(rounds):
        last_folder = str(int(sorted(rounds)[-1])+1)

    if individual_number != 1:
        last_folder = str(int(sorted(rounds)[-1]))

    if individual_number == 1 and int(last_folder) <= 5:
        os.makedirs(f'{path}/round_{last_folder}')

    # if int(last_folder) <= 5 and individual_number == 1:
    #     os.makedirs(f'src/experiments/{path}/round_{last_folder}')
    #     os.makedirs(f'src/experiments/{path}/round_{last_folder}/individual_{individual_number}')

    if int(last_folder) <=5:
        os.makedirs(f'{path}/round_{last_folder}/individual_{individual_number}')
        # os.makedirs(f'src/experiments/{path}/round_{last_folder}/individual_{individual_number}')

        with open(f'{path}/round_{last_folder}/individual_{individual_number}/train.txt', 'w') as f:
            f.write(json.dumps(final_results[f'{source}->{target}'][1]))
            f.write('\n')
            f.write(json.dumps(final_results[f'{source}->{target}'][2]))
            f.write('\n')
            f.write(json.dumps(final_results[f'{source}->{target}'][3]))
            f.close()

        with open(f'{path}/round_{last_folder}/individual_{individual_number}/test.txt', 'w') as f:
            for i in final_results[f'test:{source}->{target}']:
                f.write(json.dumps(i[0]))
                f.write('\n')
            f.close()

        with open(f'{path}/round_{last_folder}/individual_{individual_number}/refine.txt', 'w') as f:
            for i in final_results[f'refine:{source}->{target}']:
                f.write(json.dumps(i))
                f.write('\n')
            f.close()

        with open(f'{path}/round_{last_folder}/individual_{individual_number}/transfer.txt', 'w') as f:
            for i in final_results[f'transfer:{source}->{target}']:
                f.write(json.dumps(i))
                f.write('\n')
            f.close()

        with open(f'{path}/round_{last_folder}/individual_{individual_number}/inf.txt', 'w') as f:
            for i in final_results[f'inf:{source}->{target}']:
                f.write(json.dumps(i))
                f.write('\n')
            f.close()

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
