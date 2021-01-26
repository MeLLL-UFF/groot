####
# Auxiliary source
####
import re


def get_train_division(dataset):
    train_facts_source = []
    train_pos_source = []
    train_neg_source = []
    for i in range(0, len(dataset)):
        train_facts_source.extend(dataset[0][i])
        train_pos_source.extend(dataset[1][i])
        train_neg_source.extend(dataset[2][i])
    return train_facts_source, train_pos_source, train_neg_source


def get_train_test(dataset):
    train_facts = dataset[0][1:]
    train_pos = dataset[1][1:]
    train_neg = dataset[2][1:]

    test_facts = dataset[0][0]
    test_pos = dataset[1][0]
    test_neg = dataset[2][0]
    
    return train_facts, train_pos, train_neg, test_facts, test_pos, test_neg


def get_branch(curr_value, next_value):
    if curr_value == '': 
        return next_value
    return '{},{}'.format(curr_value, next_value)


def define_individual(structured_tree, tree_number):
    individual_tree = []
    target = structured_tree[0]
    nodes = structured_tree[1]
    for values, node in nodes.items():
        if values == '': 
            branch = '{} :- {}.'.format(target, node)
        else: branch = '{}.'.format(node)
        left_branch = 'true' if get_branch(values, 'true') in nodes  else 'false'
        right_branch = 'true' if get_branch(values, 'false') in nodes else 'false'
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
    new_src_struct = []
    for i in range(0, len(src_struct)): 
        new_src_struct.append(define_individual(src_struct[i], i))  
    
    return structured_src, new_src_struct


def create_pred_target(bk):
    #pred_target é : pred_target = [('movie', '+,-'), ('director', '+'),...]
    pred_target = []
    for pred in bk:
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
    

def get_refine_transfer(ind, source, target):
    refine = []
    for tree in ind.modified_src_tree:
        refine.extend(tree)
    transfer = ind.transfer.transfer
    with open(f'src/experiments/imdb_cora/refine_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(refine))
        f.close()
    with open(f'src/experiments/imdb_cora/transfer_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(transfer))
        f.close()
    return refine, transfer


def test_refine_transfer(bk, target, refine, transfer, train_dataset, test_dataset):
    background_train = boostsrl.modes(bk, [target], useStdLogicVariables=False, 
                                          maxTreeDepth=3, nodeSize=2, numOfClauses=8)
    model_tr = boostsrl.train(background_train, train_dataset[0], train_dataset[1], 
                                      train_dataset[2], refine=refine, transfer=transfer,
                                      trees=10)
    test_model = boostsrl.test(model_tr, test_dataset[0], test_dataset[1], 
                                       test_dataset[2], trees=10)
    return test_model.summarize_results(), test_model


def test_tree_b(source, target, bk_source, bk_target, transferred_structured, train_dataset, test_dataset):
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
        experiment_title = f'IMDB->cora: {source}->{target}'
        nbr = 1
        if not os.path.exists('experiments/' + experiment_title):
            os.makedirs('experiments/' + experiment_title)
        with open('experiments/' + experiment_title + '/' + str(nbr) + '_' + experiment_title + '.txt', 'a') as f:
            print(message, file=f)
            print(message)
    
    tr_file = transfer.get_transfer_file(bk_source, bk_target, predicate, to_predicate, searchArgPermutation=True, allowSameTargetMap=False)
    new_target = to_predicate
    
    # transfer and revision theory
    experiment_title = f'IMDB->cora: {source}->{target}'
    nbr = 1
    background = boostsrl.modes(bk_target, [to_predicate], useStdLogicVariables=False, maxTreeDepth=3, nodeSize=2, numOfClauses=8)
    return revision.theory_revision(background, boostsrl, target, train_dataset[0], train_dataset[1], train_dataset[2], test_dataset[0], test_dataset[1], test_dataset[2], transferred_structured, transfer=tr_file, trees=10, max_revision_iterations=1, print_function=print_function)


def save_results(final_results, source, target):
    with open(f'src/experiments/imdb_cora/groot_train_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(final_results[f'{source}->{target}']))
        f.close()
    with open(f'src/experiments/imdb_cora/groot_test_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(final_results[f'groot_test:{source}->{target}']))
        f.close()
    with open(f'src/experiments/imdb_cora/groot_inf_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(final_results[f'inf_res:{source}->{target}']))
        f.close()
    with open(f'src/experiments/imdb_cora/treeb_test_{source}_{target}.txt', 'w') as f:
        f.write(json.dumps(final_results[f'tree_test:{source}->{target}']))
        f.close()
