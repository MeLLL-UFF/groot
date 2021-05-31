from boostsrl import boostsrl
import copy
import numpy as np
import os
from random import choice, randint
import re
import shutil
import string
import sys


from src.transfer import Transfer


class Revision:

    def __init__(self):
       pass

    def remove_node(self, individual_tree, source_tree, tree_line):
        #['0;;advisedby(A, B) :- professor(A), professor(B).;true;false', 
        # '0;true;publication(C, A), publication(C, B).;false;false']
        new_individual_tree = individual_tree[0:tree_line]
        new_source_tree = source_tree[0:tree_line]
        splitted_line = individual_tree[tree_line].split(';')
        
        branch = splitted_line[1].split(',')[-1]
        parent = ','.join(splitted_line[1].split(',')[0:-1])
        parent_line = 0
        for line in range(0, tree_line):
            if individual_tree[line].split(';')[1].startswith(parent):
                parent_line = line
                break
        tmp_ind = new_individual_tree[parent_line].split(';')
        tmp_src = new_source_tree[parent_line].split(';')

        if branch == 'true':
            tmp_ind[-2] = tmp_ind[-2].replace('true', 'false')
            tmp_src[-2] = tmp_src[-2].replace('true', 'false')     
        else:
            tmp_ind[-1] = tmp_ind[-1].replace('true', 'false')
            tmp_src[-1] = tmp_src[-1].replace('true', 'false') 
        new_individual_tree[parent_line] = ';'.join(tmp_ind)  
        new_source_tree[parent_line] = ';'.join(tmp_src)  
        
        for line in range(tree_line+1, len(individual_tree)):
            if not individual_tree[line].split(';')[1].startswith(splitted_line[1]):        
                new_individual_tree.append(individual_tree[line])
                new_source_tree.append(source_tree[line])
        return new_source_tree, new_individual_tree

    def predicates(self, individual, var_list_plus, var_list_minus, number_pred):
        random_pred_target = []
        random_pred_source = []
        target_pred = individual.predicate_inst.target_pred
        for i in range(0, number_pred):
            #target
            pred = target_pred[randint(0, len(target_pred)-1)]
            num_vars = pred[1].split(',')
            chosen_variables = []
            types_source = []
            for mode in num_vars:
                if mode == '+':
                    chosen_variables.append(list(var_list_plus)[randint(0, len(var_list_plus)-1)])
                else:
                    chosen_variables.append(list(var_list_minus)[randint(0, len(var_list_minus)-1)])
                types_source.append(f'{mode}{pred[0].replace("+", "").replace("-", "").replace("_","")}type{len(types_source)+1}')
            random_pred_target.append(f'{pred[0]}({", ".join(chosen_variables)})')
            #adicionando ao kb_source
            individual.predicate_inst.kb_source.append(f'{pred[0]}target({",".join(types_source)}).')
            individual.predicate_inst.new_kb_source.append(f'{pred[0]}target({",".join(types_source)}).')
            individual.predicate_inst.new_first_kb_source.append(f'{pred[0]}target({",".join(types_source)}).')
            random_pred_source.append(f'{pred[0]}target({", ".join(chosen_variables)})')

        return ', '.join(random_pred_source), ', '.join(random_pred_target)


    def add_node(self, individual_tree, source_tree, tree_line, individual):
        variables_plus = set()
        for i in individual_tree[0:tree_line+1]:
            for char_ in i:
                if char_.isupper(): variables_plus.add(char_)

        variables_minus = set()
        for char_ in list(string.ascii_uppercase):
            if char_ not in variables_plus: variables_minus.add(char_)
                

        #nodeSize = 2
        number_pred = randint(1, 2)

        pred_source, pred_target = self.predicates(individual, variables_plus, variables_minus, number_pred)

        new_individual_tree = individual_tree[0:tree_line]
        new_source_tree = source_tree[0:tree_line]

        where_false = individual_tree[tree_line].split(';')
        where_false_src = source_tree[tree_line].split(';')
        new_node = copy.deepcopy(where_false)
        new_node_src = copy.deepcopy(where_false_src)
        if where_false[-1] == 'false':
            #indica que vai ser adicionado um novo nó no lado direito (quando der falso)
            where_false[-1] = 'true'
            where_false_src[-1] = 'true'
            if len(where_false[1]) > 0:
                new_node[1] = f'{where_false[1]},false'
                new_node_src[1] = f'{where_false_src[1]},false'
            else:
                new_node[1] = f'false'
                new_node_src[1] = f'false'
        else:
            #indica que vai ser adicionado um novo nó no lado esquerdo (quando der verdadeiro)
            where_false[-2] = 'true'
            where_false_src[-2] = 'true'
            if len(where_false[1]) > 0:
                new_node[1] = f'{where_false[1]},true'
                new_node_src[1] = f'{where_false_src[1]},true'
            else:
                new_node[1] = f'true'
                new_node_src[1] = f'true'
        new_node[-1] = f'false'
        new_node[-2] = f'false'
        new_node_src[-1] = f'false'
        new_node_src[-2] = f'false'
        new_node[2] = f'{pred_target}.'
        new_node_src[2] = f'{pred_source}.'
        
        new_individual_tree.append(';'.join(where_false))
        new_individual_tree.append(';'.join(new_node))

        new_source_tree.append(';'.join(where_false_src))
        new_source_tree.append(';'.join(new_node_src))

        for line in range(tree_line+1, len(individual_tree)):   
            new_individual_tree.append(individual_tree[line])
            new_source_tree.append(source_tree[line])
        return new_source_tree, new_individual_tree

    def leafs_to_add(self, individual_tree):
        possible_add = []
        for line in range(0, len(individual_tree)):
            splitted_line = individual_tree[line].split(';')
            if splitted_line[-1] == 'false' or splitted_line[-2] == 'false':
                possible_add.append(line)
        return possible_add

    def chose_leaf_pruning(self, individual_tree, variances):
        possible_leafs = []
        for line in range(0, len(individual_tree)):
            tree = individual_tree[line]
            if tree.endswith('false;false'):
                try:
                    var_true = variances[tree.split(';')[1]][0]
                    var_false = variances[tree.split(';')[1]][1]
                except:
                    continue
                if var_true > 0.0025 and var_false > 0.0025:
                    possible_leafs.append(line)
        return possible_leafs

    def chose_leaf_expansion(self, individual_tree, variances):
        possible_leafs = []
        for line in range(0, len(individual_tree)):
            tree = individual_tree[line].split(';')
            if tree[-1] == 'false':
                try:
                    var_true = variances[tree[1]][0]
                    if var_true > 0.0025:
                        possible_leafs.append(line)
                except:
                    pass
                # if var_true > 0.0025:
                #     possible_leafs.append(line)
            if tree[-2] == 'false':
                try:
                    var_false = variances[tree[1]][1]
                    if var_false > 0.0025:
                        possible_leafs.append(line)
                except:
                    pass
                # if var_false > 0.0025:
                #     possible_leafs.append(line)
        return possible_leafs

    def choose_node(self, individual_tree, operator, variances, random_line=True):
        if random_line:
            #escolhe nó para ser removido de forma aleatória
            if operator == 'expansion':
                return choice(self.leafs_to_add(individual_tree))
            else: #operator == pruning
                if len(individual_tree) == 1:
                    return None
                return randint(1, len(individual_tree)-1)
        else: #not random
            if operator == 'pruning':
                res = self.chose_leaf_pruning(individual_tree, variances)
                if not len(res):
                    return None
                return choice(res)
            else:
                res = self.chose_leaf_expansion(individual_tree, variances)
                if not len(res):
                    return None
                return choice(res)

    def modify_tree(self, individual, individual_tree, variances, source_tree,
                    operator, random_line=True):

        tree_line = self.choose_node(individual_tree, operator, 
                                     variances, random_line)
        if not tree_line:
            return source_tree, individual_tree

        if operator == 'expansion':
            return self.add_node(individual_tree, source_tree, tree_line, individual)
        else:
            return  self.remove_node(individual_tree, source_tree, tree_line)





