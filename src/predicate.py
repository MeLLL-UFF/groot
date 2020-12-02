import copy
from random import randint, random
import re


class Predicate:

    def __init__(self, bk_source, bk_target, pred_target):
        self.bk_source = bk_source
        self.bk_target = bk_target
        self.pred_target = pred_target
        self.new_bk_source = []
        self.new_bk_target = []
        self.variables = []
        self.mapping_var = {}

    def compare_predicates(self, bk, pred_source, pred_target):
        for pred in bk:
            if pred_target in pred:
                if len(pred.split('(')[1].split(',')) == len(pred_source.split('(')[1].split(',')):
                    return '({}'.format(pred_source.split('(')[1])
                else:
                    #COLOCAR MENSAGEM DE ERRO
                    return False
        #COLOCAR MENSAGEM DE ERRO
        return False 

    def change_predicate(self, pred, new_info, var):
        #new_info é uma lista com o que se quer colocar entre pred e as variáveis
        #pred é do tipo: predicado(A)
        new_info = "".join(str(x) for x in new_info)
        return '{}{}{}'.format(pred.split('(')[0], new_info, var)

    def generate_new_preds(self):
        self.new_bk_source = copy.deepcopy(self.bk_source)
        self.new_bk_target =copy.deepcopy(self.bk_target)

        for index in range(0, len(self.new_bk_source)):
            self.new_bk_source[index] = self.new_bk_source[index].replace('+', '')
            self.new_bk_source[index] = self.new_bk_source[index].replace('-', '')

        for index in range(0, len(self.new_bk_target)):
            self.new_bk_target[index] = self.new_bk_target[index].replace('+', '')
            self.new_bk_target[index] = self.new_bk_target[index].replace('-', '')

    def get_modes(self, individual_tree, pred_source):
        #LEVA EM CONSIDERAÇAO QUE SÓ HÁ UM PREDICADO EM CADA NÓ
        # print(pred_source, individual_tree)
        pred_source = pred_source.split('), ')
        predicate = []
        for pred in pred_source:
            if ')' not in pred:
                pred = f'{pred})'
            new_pred_source = f'{pred.split("(")[0]}('
            if not self.variables:
                self.variables.extend(("".join(re.findall('\((.*?)\)', individual_tree[0])[0])).split(', '))
            tmp_var = ("".join(re.findall('\((.*?)\)',pred))).split(',')
            self.variables = [i.strip() for i in self.variables]
            
            for var in tmp_var:
                if var.strip() in self.variables:
                    new_pred_source += '+,'
                else: 
                    new_pred_source += '-,'
                    self.variables.append(var)

            predicate.append(f'{new_pred_source})')
        return  ', '.join(predicate)

    def define_mapping(self, pred_source, pred_target):
        pred_source = pred_source.split('(')[1].split(',')
        pred_target = pred_target.split('(')[1].split(',')
        # print(pred_source, pred_target)
        for i in range(0, len(pred_source)):
            self.mapping_var[pred_source[i].replace(').', '')] = pred_target[i].replace(').', '')
        # print(self.mapping_var)

    def _get_valid_map(self, complete_source, list_pos_target):
        # print("COMPLETE SOURCE: ", complete_source)
        pred_var = []
        list_vars = complete_source.split('(')[1].split(',')
        for i in list_vars:
            tmp = i.replace(').', '')
            if tmp in self.mapping_var.keys():
                pred_var.append(self.mapping_var[tmp])
            else:
                pred_var.append('')
        # print(pred_var, len(f'({",".join(pred_var)}).'), self.mapping_var)
        # print(len(f'({",".join(pred_var)}).') == 4)
        if len(f'({",".join(pred_var)}).') == 4:
            # print("ENTREI AQUI")
            return list(range(0, len(list_pos_target)))
        else:
            pred_var = f'({",".join(pred_var)}).'
        # print(pred_var, pred_var.replace('(', ''))

        index_target = []
        for var_target in range(0, len(list_pos_target)):
            # print("PREDICADO A SER TESTADO: ")
            # print(list_pos_target[var_target])
            # print("TESTE 1: ")
            # print(list_pos_target[var_target].endswith(pred_var.replace('(', '')))
            # print("TESTE 2: ")
            # print(list_pos_target[var_target].split('(')[1].startswith(pred_var.replace(').', '').replace('(', '')))
            if list_pos_target[var_target].endswith(pred_var.replace('(', '')):
                index_target.append(var_target)
            elif list_pos_target[var_target].split('(')[1].startswith(pred_var.replace(').', '').replace('(', '')):
                index_target.append(var_target)
        if len(index_target) == 0:
            pred_var = pred_var.replace('(', '').replace(').', '')
            for index in range(0, len(list_pos_target)):
                for i in pred_var:
                    if i in list_pos_target[index]:
                        return [index]
        return index_target

    def get_valid_predicates(self, pred_source):
        #list_pred é : list_pred = [('movie', '+,-'), ('director', '+'),...]
        #pred_source é: 'movie(+person,-person).' --> NAO VEM ASSIM: verificar quais variáveis já estão na árvore, se já tiver a variável, é +, cc -
        #VERIFICAR CASO ONDE HÁ DOIS PREDICADOS NO PRED_SOURCE
        # print("PRED SOURCE: ", pred_source)

        occur_modes = ','.join([pred_source[occur.start()] 
                           for occur in re.finditer('[+\-]', pred_source)])
        valid_pred = []
        complete_valid_pred = []
        complete_pred_source = ''
        
        new_pred_source = pred_source.split('(')[0].strip()
        if ';' in new_pred_source:
            new_pred_source = new_pred_source.split(';')[2]

        for pred in self.bk_source:
            if new_pred_source in pred:
                pred_source = pred
                break
        occur_modes = ','.join([pred_source[occur.start()] 
                           for occur in re.finditer('[+\-]', pred_source)])
        # print(occur_modes)

        for pred in self.new_bk_source:
            print(pred, new_pred_source)
            if new_pred_source in pred:
                complete_pred_source = pred
                break
        for pred, mode in self.pred_target:
            if mode == occur_modes: valid_pred.append(f'{pred}({mode})')

        for target in valid_pred:
            for pred in self.new_bk_target:
                if target.split('(')[0].strip() in pred:
                    complete_valid_pred.append(pred)
                    break
        valid_index = self._get_valid_map(complete_pred_source, complete_valid_pred)
        # print("TESTE: ", pred_source, valid_index, valid_pred)
        # print("-------------------------------")
        # print("MAPPING: ", self.mapping_var)
        # print("--------------------------------")
        return complete_pred_source, [valid_pred[index] for index in valid_index], [complete_valid_pred[index] for index in valid_index]

    def get_complete_pred(self, pred, bk):
        # pred é da forma: pred(A, B) ou pred
        for i in bk:
            if pred.split('(')[0] in i:
                return i

    def new_pred(self, source_pred, target_pred):
        return f'{target_pred.split("(")[0]}({source_pred.split("(")[1]}'

    def change_pred(self, source, target, pred, pred_source):
        split_pred = pred.split(";")
        has_target = len(split_pred[2].split(":-")) > 1
        if has_target: 
            target_pred = split_pred[2].split(":- ")[1]
            # print(source)
            complete_source = self.get_complete_pred(source,self.new_bk_source)
            complete_target = self.get_complete_pred(target,self.new_bk_target)
            # print(complete_source, complete_target)
            self.define_mapping(complete_source, complete_target)
        else: target_pred = split_pred[2]

        pred_source = pred_source.split('), ')
        final_pred = []

        for pred in pred_source:
            complete_source, valid_preds, complete_valid = self.get_valid_predicates(pred)
            index_choice = randint(0, len(valid_preds)-1)
            final_pred.append(valid_preds[index_choice])
            self.define_mapping(complete_source, complete_valid[index_choice])
        if has_target: 
            split_pred[2] =  f"{split_pred[2].split(':-')[0]} :- {', '.join(final_pred)}"
        else: split_pred[2] = ', '.join(final_pred)
        return ";".join(split_pred)
