from boostsrl import boostsrl
import copy
import numpy as np
import os
from random import randint
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
            if ind_tree[line].split(';')[1].startswith(parent):
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
        return new_individual_tree, new_source_tree

    def predicates(target_pred, kb_source, var_list_plus, var_list_minus, number_pred):
        random_pred_target = []
        random_pred_source = []
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
                types_source.append(f'{mode}{pred[0]}type{len(types_source)+1}')
            random_pred_target.append(f'{pred[0]}({", ".join(chosen_variables)})')

            #adicionando ao kb_source
            kb_source.append(f'{pred[0]}target({",".join(types_source)}).')
            random_pred_source.append(f'{pred[0]}target({", ".join(chosen_variables)})')

        return ', '.join(random_pred_source), ', '.join(random_pred_target)


    def add_node(individual_tree, source_tree, tree_line, target_pred, kb_source):
        variables_plus = set()
        for i in individual_tree[0:tree_line]:
            for char_ in i:
                if char_.isupper(): variables_plus.add(char_)

        variables_minus = set()
        for char_ in list(string.ascii_uppercase):
            if char_ not in variables_plus: variables_minus.add(char_)
                
        print(len(variables_plus))

        #nodeSize = 2
        number_pred = randint(1, 2)

        pred_source, pred_target = predicates(target_pred, kb_source, variables_plus, variables_minus, number_pred)

        new_individual_tree = individual_tree[0:tree_line]
        new_source_tree = source_tree[0:tree_line]

        where_false = individual_tree[tree_line].split(';')
        new_node = copy.deepcopy(where_false)
        if where_false[-1] == 'false':
            #indica que vai ser adicionado um novo nó no lado direito (quando der falso)
            where_false[-1] = 'true'
            new_node[1] = f'{where_false[1]},false'
        else:
            #indica que vai ser adicionado um novo nó no lado esquerdo (quando der verdadeiro)
            where_false[-2] = 'true'
            new_node[1] = f'{where_false[1]},true'
        new_node[-1] = f'false'
        new_node[-2] = f'false'
        new_node[2] = f'{pred_target}.'
        print(new_node)
        new_individual_tree.append(';'.join(where_false))
        new_individual_tree.append(';'.join(new_node))

        for line in range(tree_line+1, len(individual_tree)):   
            new_individual_tree.append(individual_tree[line])
            # new_source_tree.append(source_tree[line])

        return new_individual_tree, new_source_tree

    def choose_node(self, individual_tree, random_line=True):
        #escolhe nó para ser removido de forma aleatória
        if random_line:
            return randint(1, len(individual_tree)-1)





